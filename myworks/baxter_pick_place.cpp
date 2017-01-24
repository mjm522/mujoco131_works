#include "mujoco_viewer.cpp"


//-------------------------------- main function ----------------------------------------

int main(int argc, const char** argv)
{
    // print version, check compatibility
	printf("MuJoCo Pro library version %.2lf\n\n", 0.01*mj_version());
	if( mjVERSION_HEADER!=mj_version() )
		mju_error("Headers and library have different versions");

	// activate MuJoCo license
	mj_activate("../bin/mjkey.txt");
	
    // init GLFW, set multisampling
    if (!glfwInit())
        return 1;
    glfwWindowHint(GLFW_SAMPLES, 4);

    // try stereo if refresh rate is at least 100Hz
    GLFWwindow* window = 0;
    if( glfwGetVideoMode(glfwGetPrimaryMonitor())->refreshRate>=100 )
    {
        glfwWindowHint(GLFW_STEREO, 1);
        window = glfwCreateWindow(1200, 900, "Simulate", NULL, NULL);
        if( window )
            stereoavailable = true;
    }

    // no stereo: try mono
    if( !window )
    {
        glfwWindowHint(GLFW_STEREO, 0);
        window = glfwCreateWindow(1200, 900, "Simulate", NULL, NULL);
    }
    if( !window )
    {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    // determine retina scaling
    int width, width1, height;
    glfwGetFramebufferSize(window, &width, &height);
    glfwGetWindowSize(window, &width1, &height);
    scale = (double)width/(double)width1;

    // init MuJoCo rendering
    mjv_makeObjects(&objects, 1000);
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&vopt);
    mjr_defaultOption(&ropt);
    mjr_defaultContext(&con);
    mjr_makeContext(m, &con, 200);

    // load model from xml file
    loadmodel(window, "../models/baxter/baxter_pad_ee.xml", 0);
    
    // set GLFW callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
    glfwSetDropCallback(window, drop);

    //max duration
    int itr = 0, maxItr = 10000;

    double qpos[21]    = {0};
    
    double untuck_l[7] = {-0.08, -1.0,  -1.19, 1.94,  0.67, 1.03, -0.50};
    double untuck_r[7] = {0.08, -1.0,   1.19, 1.94,  -0.67, 1.03,  0.50};

    get_state(qpos, m, d);

    mju_copy(&qpos[1], untuck_r,  7);
    mju_copy(&qpos[8], untuck_l,  7);

    set_state(qpos, m, d);
    
    // main loop
    while( !glfwWindowShouldClose(window))
    {
        itr++;
        // simulate and render
        render(window, itr);

        // finalize
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // delete everything we allocated
    mj_deleteData(d);
    mj_deleteModel(m);
    mjr_freeContext(&con);
    mjv_freeObjects(&objects);

    // terminate
    glfwTerminate();
	mj_deactivate();
    return 0;
}