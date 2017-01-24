#include "mujoco.h"
#include <algorithm> 
#include "Eigen/Dense"

// model
mjModel* m = 0;
mjData* d = 0;
char lastfile[1000] = "";

using namespace Eigen;

void get_state(mjtNum* ptr, const mjModel* m, const mjData* d) 
{
    mju_copy(ptr, d->qpos, m->nq);
}

void set_state(const mjtNum* ptr, const mjModel* m, mjData* d) 
{
    mju_copy(d->qpos, ptr, m->nq);
}

void quatdiff(mjtNum* quat_diff, mjtNum* quat_curr, mjtNum* quat_des)
{    
    mjtNum tmp[3];
    
    for (int i=1; i<4; i++)
        quat_diff[i] = quat_curr[i]*quat_des[0] - quat_des[i]*quat_curr[0];
   
    mju_cross(tmp, quat_des+1, quat_curr+1);
    mju_addTo(quat_diff, tmp, 3);
}

void inverse_dynamics(mjtNum* tau, mjtNum* qpos, mjtNum* qvel, mjtNum* qacc)
{
    mjData* d_tmp = 0;
    d_tmp = mj_makeData(m);
    mju_copy(d->qpos, qpos, m->nq);
    mju_copy(d_tmp->qvel, qvel, m->nv);
    mju_copy(d->qacc,     qacc, m->nv);
    mju_copy(tau, d->qfrc_inverse, m->nv);
    mj_deleteData(d_tmp);
}


// simple controller applying damping to each dof
void mycontroller(const mjModel* m, mjData* d)
{
    mj_step1(m, d);

    /*

    // initialization
    Matrix<mjtNum, 3, 1> test_site_pos;
    Matrix<mjtNum, 4, 1> test_site_ori;

    //left hand
    Matrix<mjtNum, 3, 1> l_ee_pos;
    Matrix<mjtNum, 4, 1> l_ee_ori;
    Matrix<mjtNum, 6, 1> l_ee_vel;
    Matrix<mjtNum, 6, 1> l_ee_acc;
    Matrix<mjtNum, 15, 1> l_joint_pos;
    Matrix<mjtNum, 15, 1> l_joint_vel;
    Matrix<mjtNum, 15, 1> l_joint_acc;
    Matrix<mjtNum, 15, 1> l_joint_vel_req;

    Matrix<mjtNum, 225, 1> Mass_Matrix_tmp;

    //right hand
    Matrix<mjtNum, 3, 1> r_ee_pos;
    Matrix<mjtNum, 4, 1> r_ee_ori;
    Matrix<mjtNum, 6, 1> r_ee_vel;
    Matrix<mjtNum, 6, 1> r_ee_acc;
    Matrix<mjtNum, 15, 1> r_joint_pos;
    Matrix<mjtNum, 15, 1> r_joint_vel;
    Matrix<mjtNum, 15, 1> r_joint_acc;
    Matrix<mjtNum, 15, 1> r_joint_vel_req;

    //space for jacobian, m->nv = 15
    Matrix<mjtNum, 90, 1> l_ee_jac_tmp;
    Matrix<mjtNum, 90, 1> r_ee_jac_tmp;

    //space for pseudo inverse
    Matrix<mjtNum, 15, 6> l_ee_jac_pinv;
    Matrix<mjtNum, 15, 6> r_ee_jac_pinv;

    //space for errors
    //position
    Matrix<mjtNum, 6, 1> err_l_ee;     err_l_ee.setZero();
    Matrix<mjtNum, 6, 1> err_r_ee;     err_r_ee.setZero();

    //space for final results
    Matrix<mjtNum, 15, 1> qpos;        qpos.setZero();
    Matrix<mjtNum, 15, 1> qvel;        qvel.setZero();
    Matrix<mjtNum, 15, 1> qacc;        qacc.setZero();

    //a test site in the xml file
    int test_site_id = mj_name2id(m, mjOBJ_SITE, "testSite");
    //this is the tip point of the left arm
    int l_ee_site_id = mj_name2id(m, mjOBJ_SITE, "left_pad_center");
    // this is the tip point of right arm
    int r_ee_site_id = mj_name2id(m, mjOBJ_SITE, "right_pad_center");

    //test site
    mju_copy(test_site_pos.data(), &d->site_xpos[test_site_id],  3);
    mju_copy(test_site_ori.data(), &d->site_xmat[test_site_id],  4);

    //left arm ee
    mju_copy(l_ee_pos.data(), &d->site_xpos[l_ee_site_id],  3);
    mju_copy(l_ee_ori.data(), &d->site_xmat[l_ee_site_id],  4);

    //right arm ee
    mju_copy(r_ee_pos.data(), &d->site_xpos[r_ee_site_id],  3);
    mju_copy(r_ee_ori.data(), &d->site_xmat[r_ee_site_id],  4);

    //compute jacobian of left and right hand
    mj_jacSite(m, d, l_ee_jac_tmp.data(), l_ee_jac_tmp.data()+45, l_ee_site_id);
    mj_jacSite(m, d, r_ee_jac_tmp.data(), r_ee_jac_tmp.data()+45, r_ee_site_id);

    Matrix<mjtNum, 6, 15> l_ee_jac(l_ee_jac_tmp.data());
    Matrix<mjtNum, 6, 15> r_ee_jac(r_ee_jac_tmp.data());

    //position error left and right hand
    for (int i=0; i<3; i++)
    {
        err_l_ee(i,0) =  test_site_pos(i,0) - l_ee_pos(i,0);
        err_r_ee(i,0) =  test_site_pos(i,0) - r_ee_pos(i,0);
    }

    //right hand orientation error
    quatdiff(err_l_ee.data()+3, l_ee_ori.data(), test_site_ori.data());

    //left hand orientation error
    quatdiff(err_r_ee.data()+3, r_ee_ori.data(), test_site_ori.data());

    l_ee_jac_pinv = l_ee_jac.transpose() * (l_ee_jac * l_ee_jac.transpose()).inverse();
    r_ee_jac_pinv = r_ee_jac.transpose() * (r_ee_jac * r_ee_jac.transpose()).inverse();

    //required joint velocities
    l_joint_vel = l_ee_jac_pinv * err_l_ee;
    r_joint_vel = r_ee_jac_pinv * err_r_ee;

    //current position, velocity
    mju_copy(qpos.data(), d->qpos,  15);
    mju_copy(qvel.data(), d->qvel,  15);

    //required joint accelerations
    l_joint_acc =  l_joint_vel - qvel;
    r_joint_acc =  r_joint_vel - qvel;

    mj_fullM(m, Mass_Matrix_tmp.data(), d->qM);

    Matrix<mjtNum, 15, 15> Mass_Matrix(Mass_Matrix_tmp.data());    

    mju_copy(qacc.data()+1,  r_joint_acc.data()+1,   7); // copying relevant data for right hand
    mju_copy(qacc.data()+9,  l_joint_acc.data()+9,   7); // copying relevant data for left hand

    */


    Matrix<mjtNum, 15, 1> tau; tau.setZero();


    /////////////////////////////////THIS DOESN'T WORK/////////////////////////////

    //inverse_dynamics(tau.data(), qpos.data(), qvel.data(), qacc.data());

    ///////////////////////////////////////////////////////////////////////////////


    /////////////////****************** APPROXIMATELY WORKS****************//////////////////////////
    //copying acceleration directly to tau and scaling it.
    //tau = Mass_Matrix*qacc;
    //tau = qacc;

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    
    
    //adding bias terms to tau
    mju_addTo(tau.data(), d->qfrc_bias, 15);
    mju_addTo(tau.data(), d->qfrc_passive, 15);

    // mju_printMat(tau.data(), 1, 15);

    mju_copy(d->ctrl, tau.data(), m->nu);

    // mju_scl(d->qfrc_applied, d->qfrc_bias, -1, 15);

    

    
    mj_step2(m, d);

}