// mujoco communication interface
#include "comm.h"

// algebra classes
#include "GMatrix.h"
#include "GVector.h"
#include "Quaternion.h"
#include "windows.h"

typedef GVector<double> Vector;
typedef GMatrix<double> Matrix;
typedef Quaternion<double> Quat;

// sets thetahat such that gripper fingers open or close by gripAmount
// gripAmount = 0: default gipper posture
// gripAmount > 0: gripper fingers opened
// gripAmount < 0: gripper fingers closed
// between -1 and +1 are acceptable gripAmount values
void setGrip(double gripAmount, Vector &thetahat)
{
  // indices for joints of two gripper fingers
  static const int L_GRIP_JOINT_INDEX = 7;
  static const int R_GRIP_JOINT_INDEX = 8;
  
  thetahat[L_GRIP_JOINT_INDEX] = -gripAmount;
  thetahat[R_GRIP_JOINT_INDEX] = +gripAmount;
}

// Compute delta theta (the change in target arm settings) using the Theta
// Transpose method.
// @param alpha The alpha factor used to scale the delta (<= 1.0).
// @param delta The delta vector showing the difference we're trying to reduce.
// @param jacobian The jacobian matrix for the arm.
Vector ComputeDeltaThetaTranpose(double alpha, const Vector& delta, const Matrix& jacobian) {
	// The Jacobian Transpose Method from slide 7:
	return jacobian.transpose() * alpha * delta;
}

// Compute delta theta (the change in target arm settings) using the Pseudo
// Inverse method.
// @param alpha The alpha factor used to scale the delta (<= 1.0).
// @param delta The delta vector showing the difference we're trying to reduce.
// @param jacobian The jacobian matrix for the arm.
// @param theta The current arm settings.
// @param thetaReference The reference position for the arm.
// @param I An identity matrix that matches the size of the jacobian matrix.
Vector ComputeDeltaThetaPseudoInverse(double alpha, const Vector& delta, const Matrix& jacobian,
									  const Vector& theta, const Vector& thetaReference,
									  const Matrix& I) {
	// The Pseudo Inverse with Explicit Optimization Criterion approach from slide 13:
	Matrix jPound = jacobian.transpose() * (jacobian * jacobian.transpose()).inverse();
	return jPound * alpha * delta + (I - jPound * jacobian) * (thetaReference - theta);
}

// Compute the updated theta (the target arm setting).  This is based on the
// existing arm settings plus a delta, determined either by the Jacobian
// Transpose method or the Pseudo Inverse one.
//
// @param alpha The alpha factor used to scale the delta (<= 1.0).
// @param x The position vector for the arm endpoint.
// @param xhat The position vector for the arm endpoint's target location.
// @param r The rotation of the arm endpoint.
// @param rhat The rotation of the arm endpoint's target location.
// @param Jpos Positional jacobian matrix.
// @param Jrot Rotational jacobian matrix.
// @param theta The current arm settings.
// @param thetaReference The reference position for the arm.
// @param I An identity matrix that matches the size of the jacobian matrix.
Vector ComputeCombinedThetaHat(double alpha, const Vector& x, const Vector& xhat,
							   const Quat& r, const Quat& rhat,
							   const Matrix& Jpos, const Matrix& Jrot,
							   const Vector& theta, const Vector& thetaReference,
							   const Matrix& I) {
	// Build positional and rotational deltas, and merge them into deltaCombined.
	Vector deltaPos = xhat - x;
	Vector deltaRot = quatdiff(r, rhat);
	double newVec[] = {deltaPos[0], deltaPos[1], deltaPos[2],
						deltaRot[0], deltaRot[1], deltaRot[2]};
	Vector deltaCombined(6, newVec);

	// Build the combined/contatenated Jacobian matrix.
	Matrix combinedJacobian(6, Jpos.getNumCols());
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < Jpos.getNumCols(); ++j) {
			combinedJacobian[i][j] = Jpos[i][j];
			combinedJacobian[i + 3][j] = Jrot[i][j];
		}
	}

	// Use these flags to determine which method is used to compute delta theta.
	static const bool part3JacobianTranspose = false;
	static const bool part3PseudoInverse = true;
	if (part3JacobianTranspose) {
		Vector deltaTheta = ComputeDeltaThetaTranpose(alpha, deltaCombined, combinedJacobian);
		return theta + deltaTheta;
	} else if (part3PseudoInverse) {
		return theta + ComputeDeltaThetaPseudoInverse(alpha, deltaCombined, combinedJacobian, theta,
			thetaReference, I);
	}
	return theta;
}

// This class stores all the state needed to implement Part 4 of the homework.
// It implements a state machine to move toward the capsule, move in and grab it,
// move it to the target, then release it and move the arm away.
class Part4 {
public:
	Part4() : stage_(MOVE_BESIDE_CAPSULE), waitUpdate_(0), updatesToDone_(0) {}

	// Update function that's called once per frame to update the arm.  As the
	// exit criteria for each state of the state machine are met, this moves from
	// one state to the next.
	//
	// @param x The position vector for the arm endpoint.
	// @param xhat The position vector for the arm endpoint's target location.
	// @param r The rotation of the arm endpoint.
	// @param rhat The rotation of the arm endpoint's target location.
	// @param Jpos Positional jacobian matrix.
	// @param Jrot Rotational jacobian matrix.
	// @param theta The current arm settings.
	// @param thetaReference The reference position for the arm.
	// @param I An identity matrix that matches the size of the jacobian matrix.
	// @param forearmDirection A positional vector roughly indicating the forearm direction.
	// @param objectPosition Position vector for the capsule we're moving.
	// @param objectRotation Rotation of the capsule we're moving.
	Vector Update(const Vector& x, const Vector& xhat,
			      const Quat& r, const Quat& rhat,
			      const Matrix& Jpos, const Matrix& Jrot,
			      const Vector& theta, const Vector& thetaReference,
			      const Matrix& I, const Vector& forearmDirection,
				  const Vector& objectPosition, const Quat& objectRotation) {
		static const double clawClosedValue = -0.25;
		++updatesToDone_;

		switch (stage_) {
			case MOVE_BESIDE_CAPSULE:
				// Move the arm so it's close to the capsule and open the claw.  This is
				// done first so we make sure the claw is at a good orientation to move
				// toward the capsule.
				{
					Vector thetahat = MoveToOffsetPosition(0.01, 0.12, MOVE_CLOSER, forearmDirection, objectPosition, objectRotation,
						x, r, Jpos, Jrot, theta, thetaReference, I);
					// Open the claw all the way.
					setGrip(1.0, thetahat);
					return thetahat;
				}

			case MOVE_CLOSER:
				// Move the claw closer to the capsule, so it's in a position to grab it.
				{
					Vector thetahat = MoveToOffsetPosition(0.01, 0.06, GRAB_CAPSULE, forearmDirection, objectPosition, objectRotation,
						x, r, Jpos, Jrot, theta, thetaReference, I);
					// Open the claw all the way.
					setGrip(1.0, thetahat);
					return thetahat;
				}

			case GRAB_CAPSULE:
				// Close the claw and grab the capsule.
				{
					// Close the enough that it grabs the capsule.
					Vector thetahat = theta;
					setGrip(clawClosedValue, thetahat);

					static const int L_GRIP_JOINT_INDEX = 7;
					static const int R_GRIP_JOINT_INDEX = 8;
					if (abs(thetahat[L_GRIP_JOINT_INDEX] - theta[L_GRIP_JOINT_INDEX]) < 0.01) {
						++waitUpdate_;
						if (waitUpdate_ > 10) {
							stage_ = MOVE_TO_TARGET;
							waitUpdate_ = 0;
						}
					}

					return thetahat;
				}

			case MOVE_TO_TARGET:
				// Move the arm so the capsule's position and orientation match the
				// target.
				{
					// Move based on the object's position and rotation, since it's now attached to the arm.
					Vector thetahat = ComputeCombinedThetaHat(0.001, objectPosition, xhat,
						objectRotation, rhat, Jpos, Jrot, theta, thetaReference, I);
					// Keep the claw closed.
					setGrip(clawClosedValue, thetahat);

					// In theory, we should wait for both position and orientation to be aligned.
					// In practice, it's enough to wait for the position to be aligned, because the
					// orientation is fine by the time the position is good.
					if ((objectPosition - xhat).length() < 0.001) {
						stage_ = RELEASE_CAPSULE;
					}

					return thetahat;
				}

			case RELEASE_CAPSULE:
				{
					// Open the claw.
					Vector thetahat = theta;
					setGrip(1.0, thetahat);

					static const int L_GRIP_JOINT_INDEX = 7;
					static const int R_GRIP_JOINT_INDEX = 8;
					if (abs(thetahat[L_GRIP_JOINT_INDEX] - theta[L_GRIP_JOINT_INDEX]) < 0.01) {
						++waitUpdate_;
						if (waitUpdate_ > 10) {
							stage_ = MOVE_AWAY;
							waitUpdate_ = 0;
						}
					}

					return thetahat;
				}

			case MOVE_AWAY:
				// Move away from the capsule and leave it at the target.
				{
					double awayAxisValues[] = { -1.0,  0.0, 0.0 };
					Vector away = r.rotate(Vector(3, awayAxisValues));
					Vector updatedTargetPosition = xhat + away * 0.12;
					Vector thetahat = ComputeCombinedThetaHat(0.01, x, updatedTargetPosition,
						r, r, Jpos, Jrot, theta, thetaReference, I);
					// Open the claw all the way.
					setGrip(1.0, thetahat);

					if ((x - updatedTargetPosition).length() < 0.001) {
						stage_ = DONE;
						printf("Finished after %d updates.\n", updatesToDone_);
					}

					return thetahat;
				}

			case DONE:
				// At this point, we're done.  The arm will sit here and do nothing.
				// If the world is reset, go back to the first stage.
				if ((objectPosition - xhat).length() > 0.1) {
					++waitUpdate_;
					if (waitUpdate_ > 10) {
						stage_ = MOVE_BESIDE_CAPSULE;
						waitUpdate_ = 0;
						updatesToDone_ = 0;
					}
				}
				break;
		}
		return theta;
	}

private:
	enum Stages {
		MOVE_BESIDE_CAPSULE,
		MOVE_CLOSER,
		GRAB_CAPSULE,
		MOVE_TO_TARGET,
		RELEASE_CAPSULE,
		MOVE_AWAY,
		DONE
	};

	// Calculate a target rotation that's rotated 90 degrees from the object's
	// current rotation, around the provided vector.
	//
	// @param away A unit vector providing the axis of rotation.
	// @param objectRotation The target object's current rotation.
	Quat CalcTargetRotation(const Vector& away, const Quat& objectRotation) {
		Quat clawRotation90(3.14159265 / 2, away);
		return clawRotation90 * objectRotation;
	}

	// Move the arm to a position that's near the target, but offset by a provided
	// amount, away from whatever direction the forearm is pointing.
	//
	// @param alpha The alpha value used to scale delta theta.
	// @param awayDist Distance away from the capsule for our target position.
	// @param nextStage Once we're in position, change the state machine to this stage.
	// @param forearmDirection A positional vector roughly indicating the forearm direction.
	// @param targetPosition Position vector for the target we're moving toward.
	// @param targetRotation Rotation of the target we're moving toward.
	// @param x The position vector for the arm endpoint.
	// @param r The rotation of the arm endpoint.
	// @param Jpos Positional jacobian matrix.
	// @param Jrot Rotational jacobian matrix.
	// @param theta The current arm settings.
	// @param thetaReference The reference position for the arm.
	// @param I An identity matrix that matches the size of the jacobian matrix.
	Vector MoveToOffsetPosition(double alpha, double awayDist, Stages nextStage,
		const Vector& forearmDirection, const Vector& targetPosition,
		const Quat& targetRotation, const Vector& x, const Quat& r, const Matrix& Jpos,
		const Matrix& Jrot, const Vector& theta, const Vector& thetaReference, const Matrix& I)
	{
		// Move to a position that's slightly away from the capsule, opposite the direction of the forearm.
		Vector away = -forearmDirection;
		away.normalize();
		Vector updatedTargetPosition = targetPosition + away * awayDist;

		// Rotate the target rotation 90 degrees.
		Quat updatedTargetRotation = CalcTargetRotation(away, targetRotation);

		// If we're close enough to the target position, move to the next stage.
		// This could also check rotation, but rotation seems to align faster than
		// position in all states of this state machine, so there's no need.  Plus,
		// since the capsule and the target have one axis of rotation that they don't
		// care about, it's simpler to leave that check out.
		if ((updatedTargetPosition - x).length() < 0.001) {
			stage_ = nextStage;
		}

		return ComputeCombinedThetaHat(alpha, x, updatedTargetPosition,
			r, updatedTargetRotation, Jpos, Jrot, theta, thetaReference, I);
	}

	Stages stage_;
	int waitUpdate_;
	int updatesToDone_; // Number of update calls before we reached the Done stage.
};

void main(void)
{
  // indices for nodes in the scene
	static const int TARGET_GEOM_INDEX = 1;
	static const int LOWER_ARM_GEOM_INDEX = 3;
	static const int HAND_GEOM_INDEX = 4;
	static const int OBJECT_GEOM_INDEX = 13; 

	// connect to mujoco server
	mjInit();
	mjConnect(10000);
	Part4 part4;

	// load hand model
	if(mjIsConnected())
	{
		mjLoad("hand.xml");
		Sleep(1000);  // wait till the load is complete
	}

	if(mjIsConnected() && mjIsModel())
	{
		mjSetMode(2);
		mjReset();

		// size containts model dimensions
		mjSize size = mjGetSize();

		// number of arm degress of freedom (DOFs) and controls
		// (does not include target object degrees of freedom)
		int dimtheta = size.nu;

		// identity matrix (useful for implementing Jacobian pseudoinverse methods)
		Matrix I(dimtheta, dimtheta);
		I.setIdentity();

		// target arm DOFs
		Vector thetahat(dimtheta);
		thetahat.setConstant(0.0);

		Vector thetaReference(dimtheta);
		thetaReference.setConstant(0.0);

		int frameNum = 0;
		bool done = false;
		Vector minPos(3), maxPos(3);
		for(;;) // run simulation forever
		{
			// simulation advance substep
			mjStep1();

			mjState state = mjGetState();
			// current arm degrees of freedom
			Vector theta(dimtheta);
			// state.qpos contains DOFs for the whole system. Here extract
			// only DOFs for the arm into theta
			bool reset = true;
			for(int e=0; e<dimtheta; e++)
			{
				theta[e] = state.qpos[e];
				if (theta[e] != 0) {
					reset = false;
				}
			}
			if (reset) {
				printf("Reset.\n");
				frameNum = 0;
				done = false;
			} else {
				++frameNum;
			}

			mjCartesian geoms = mjGetGeoms();
			// Lower arm position
			Vector lowerArmPos(3, geoms.pos + 3*LOWER_ARM_GEOM_INDEX);
			// current hand position
			Vector x(3, geoms.pos + 3*HAND_GEOM_INDEX);
			// target hand position
			Vector xhat(3, geoms.pos + 3*TARGET_GEOM_INDEX);
			// current hand orientation
			Quat r(geoms.quat + 4*HAND_GEOM_INDEX);
			// target hand orientation
			Quat rhat(geoms.quat + 4*TARGET_GEOM_INDEX);
			// object position
			Vector objectPosition(3, geoms.pos + 3*OBJECT_GEOM_INDEX);
			// object orientation
			Quat objectRotation(geoms.quat + 4*OBJECT_GEOM_INDEX);

			if (reset) {
				minPos = maxPos = x;
			} else {
				for (int i = 0; i < 3; ++i) {
					minPos[i] = min(minPos[i], x[i]);
					maxPos[i] = max(maxPos[i], x[i]);
				}
			}

			mjJacobian jacobians = mjJacGeom(HAND_GEOM_INDEX);
			// current hand position Jacobian
			Matrix Jpos(3,jacobians.nv, jacobians.jacpos);
			// current hand orientation Jacobian
			Matrix Jrot(3,jacobians.nv, jacobians.jacrot);
			// extract only columns of the Jacobian that relate to arm DOFs
			Jpos = Jpos.getBlock(0,3, 0, dimtheta);
			Jrot = Jrot.getBlock(0,3, 0, dimtheta);

			// -- your code goes here --
			// Implement part 1.  Use these constants to adjust which method is used and
			// what alpha value to use.  If both bools are false, Part 1 is skipped.
			static const double alpha = 0.01;
			static const bool part1JacobianTranspose = false;
			static const bool part1PseudoInverse = false;
			if (!done && (part1JacobianTranspose || part1PseudoInverse)) {
				if (part1JacobianTranspose) {
					thetahat = theta + ComputeDeltaThetaTranpose(alpha, xhat - x, Jpos);
				} else if (part1PseudoInverse) {
					thetahat = theta + ComputeDeltaThetaPseudoInverse(alpha, xhat - x, Jpos, theta,
						thetaReference, I);
				}
				// Once we reach the target, display some information about how long it
				// took and how much the arm moved around.
				if ((xhat - x).length() < 0.001) {
					done = true;
					printf("Finished after %d frames.\n", frameNum);
					printf("Bounding box <%lf, %lf, %lf>.\n", maxPos[0] - minPos[0],
						maxPos[1] - minPos[1], maxPos[2] - minPos[2]);
				}
			}

			// Calculate orientation jacobian.  Set one of these flags to true to
			// try out Part 2.
			static const bool part2JacobianTranspose = false;
			static const bool part2PseudoInverse = false;
			if (!done && (part2JacobianTranspose || part2PseudoInverse)) {
				if (part2JacobianTranspose) {
					thetahat = theta + ComputeDeltaThetaTranpose(alpha, quatdiff(r, rhat), Jrot);
				} else if (part2PseudoInverse) {
					thetahat = theta + ComputeDeltaThetaPseudoInverse(alpha, quatdiff(r, rhat), Jrot, theta,
						thetaReference, I);
				}
				// Once we reach the target, display some information about how long it
				// took and how much the arm moved around.
				if (quatdiff(r, rhat).length() < 0.001) {
					done = true;
					printf("Finished after %d frames.\n", frameNum);
					printf("Bounding box <%lf, %lf, %lf>.\n", maxPos[0] - minPos[0],
						maxPos[1] - minPos[1], maxPos[2] - minPos[2]);
				}
			}

			// Part 3, combine position and orientation.  Set doPart3 to true to
			// run this part of the code.
			static const bool doPart3 = false;
			if (!done && doPart3) {
				thetahat = ComputeCombinedThetaHat(alpha, x, xhat, r, rhat, Jpos, Jrot, theta, thetaReference, I);
				// Once we reach the target, display some information about how long it
				// took and how much the arm moved around.
				if ((xhat - x).length() < 0.001) {
					done = true;
					printf("Finished after %d frames.\n", frameNum);
					printf("Bounding box <%lf, %lf, %lf>.\n", maxPos[0] - minPos[0],
						maxPos[1] - minPos[1], maxPos[2] - minPos[2]);
				}
			}

			// Set doPart4 to true to run the code for Part 4.
			static const bool doPart4 = true;
			if (doPart4) {
				thetahat = part4.Update(x, xhat, r, rhat, Jpos, Jrot, theta, thetaReference, I,
					x - lowerArmPos, objectPosition, objectRotation);
			}

			// set target DOFs to thetahat and advance simulation
			mjSetControl(dimtheta, thetahat);
			mjStep2();
		}

		mjDisconnect();
	}
	mjClear();
}
