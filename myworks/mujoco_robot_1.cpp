#include "mujoco.h"
#include <algorithm> 

// model
mjModel* m = 0;
mjData* d = 0;
char lastfile[1000] = "";

int state_size(mjModel* m) 
{
    return m->nq + m->nv;
}

void get_vel(mjtNum* ptr, const mjModel* m, const mjData* d) 
{
    mju_copy(ptr, d->qvel, m->nv);
}

void get_state(mjtNum* ptr, const mjModel* m, const mjData* d) 
{
    mju_copy(ptr, d->qpos, m->nq);
}

void get_bias_force(mjtNum* ptr, const mjModel* m, const mjData* d)
{ 
    std::fill_n(ptr, m->nv, 0); 
    mju_addTo(ptr, d->qfrc_bias, m->nv);
    //mju_addTo(ptr, d->qfrc_passive, m->nv);
}

void set_vel(const mjtNum* ptr, const mjModel* m, mjData* d) 
{
    mju_copy(d->qvel, ptr, m->nv);
}

void set_state(const mjtNum* ptr, const mjModel* m, mjData* d) 
{
    mju_copy(d->qpos, ptr, m->nq);
}


void set_ctrl(const mjtNum* ptr, const mjModel* m, mjData* d) 
{
    mju_copy(d->ctrl, ptr, m->nu);
}


void site_pose(mjtNum* site_pos, mjtNum* site_ori, const mjData* d, const char* site_name)
{
	
    int site_id = mj_name2id(m, mjOBJ_SITE, site_name);
    mju_copy(site_pos, &d->site_xpos[site_id],  3);
    mju_copy(site_ori, &d->site_xmat[site_id],  4);

}

void site_jacobian(mjtNum* site_pos_jacobian, mjtNum* site_ori_jacobian, const mjData* d, const char* site_name)
{
    //this function assumes that the site is on the robot.
    int site_id = mj_name2id(m, mjOBJ_SITE, site_name);

    double* jac_pos_tmp = new double[3*m->nv];
    double* jac_ori_tmp = new double[3*m->nv];

    mj_jacSite(m, d, jac_pos_tmp, jac_ori_tmp, site_id);

    mju_copy(site_pos_jacobian, jac_pos_tmp,  3*m->nu);
    mju_copy(site_ori_jacobian, jac_ori_tmp,  3*m->nu);

    delete[] jac_pos_tmp;
    delete[] jac_ori_tmp;
}


// simple controller applying damping to each dof
void mycontroller(const mjModel* m, mjData* d)
{
    // mj_step1(m,d);

    double *qfrc_bias = new double[m->nv];
    
    double bx_l_site_pos[3] = {0};
    double bx_l_site_ori[4] = {0};
    double bx_r_site_pos[3] = {0};
    double bx_r_site_ori[4] = {0};

    double l_ee_pos[3] = {0};
    double l_ee_ori[4] = {0};
    double r_ee_pos[3] = {0};
    double r_ee_ori[4] = {0};

    double l_ee_pos_jac[45] = {0};
    double l_ee_ori_jac[45] = {0};

    double r_ee_pos_jac[45] = {0};
    double r_ee_ori_jac[45] = {0};

    site_pose(bx_l_site_pos, bx_l_site_ori, d, "f2");
    site_pose(bx_r_site_pos, bx_r_site_ori, d, "f4");
    site_pose(l_ee_pos, l_ee_ori, d, "left_pad_center");
    site_pose(r_ee_pos, r_ee_ori, d, "right_pad_center");

    site_jacobian(l_ee_pos_jac, l_ee_ori_jac, d, "left_pad_center");
    site_jacobian(r_ee_pos_jac, r_ee_ori_jac, d, "right_pad_center");

    double bx_r_site_pos_delta[3] = {0};
    double bx_l_site_pos_delta[3] = {0};
    double l_cmd_tmp[15]          = {0};
    double r_cmd_tmp[15]          = {0};

    mju_sub3(bx_l_site_pos_delta, l_ee_pos, bx_l_site_pos);
    mju_sub3(bx_r_site_pos_delta, r_ee_pos, bx_r_site_pos);

    mju_transpose(r_ee_pos_jac, r_ee_pos_jac, 3, 15);
    mju_transpose(l_ee_pos_jac, l_ee_pos_jac, 3, 15);

    mju_mulMatVec(r_cmd_tmp, r_ee_pos_jac, bx_r_site_pos_delta, 15, 3);
    mju_mulMatVec(l_cmd_tmp, l_ee_pos_jac, bx_l_site_pos_delta, 15, 3);

    double* cmd = new double[m->nu];

    std::fill_n(cmd, m->nu, 0); 

    mju_copy(&cmd[1], r_cmd_tmp,      7);
    mju_copy(&cmd[8], &l_cmd_tmp[8],  7);

    std::cout <<"left error \n";
    mju_printMat(bx_l_site_pos_delta, 1, 3);
    std::cout <<"\n left delta cmd \n";
    mju_printMat(l_cmd_tmp, 1, 7);
    std::cout <<" \n right error \n";
    mju_printMat(bx_r_site_pos_delta, 1, 3);
    std::cout <<"\n right delta cmd \n";
    mju_printMat(r_cmd_tmp, 1, 7);

    // mju_addTo(cmd, d->qpos, m->nu);
    get_bias_force(qfrc_bias, m, d);
    //add only the ones that correspond to joints
    mju_addTo(cmd, qfrc_bias, m->nu);
   
    set_ctrl(cmd, m, d);
    mj_step(m,d); 

    delete[] cmd;
    delete[] qfrc_bias;
     
}