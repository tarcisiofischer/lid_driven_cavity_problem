#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <omp.h>

namespace py = pybind11;
typedef double __real_t;
typedef long long __integer_t;

py::array residual_function_omp(py::array X, py::object graph)
{
    auto residual = py::array_t<__real_t>(X.size());
    auto is_residual_calculated = py::array_t<bool>(X.size());
    
    // Extract useful variables from python objects
    auto residual_size = residual.size();
    auto pressure_mesh_size = (__integer_t)py::len(graph.attr("pressure_mesh"));
    auto pressure_mesh_nx = (__integer_t)py::int_(graph.attr("pressure_mesh").attr("nx"));
    auto pressure_mesh_ny = (__integer_t)py::int_(graph.attr("pressure_mesh").attr("ny"));
    auto ns_x_mesh_size = (__integer_t)py::len(graph.attr("ns_x_mesh"));
    auto ns_x_mesh_nx = (__integer_t)py::int_(graph.attr("ns_x_mesh").attr("nx"));
    auto ns_x_mesh_ny = (__integer_t)py::int_(graph.attr("ns_x_mesh").attr("ny"));
    auto ns_y_mesh_size = (__integer_t)py::len(graph.attr("ns_y_mesh"));
    auto ns_y_mesh_nx = (__integer_t)py::int_(graph.attr("ns_y_mesh").attr("nx"));
    auto ns_y_mesh_ny = (__integer_t)py::int_(graph.attr("ns_y_mesh").attr("ny"));
    auto *residual_ptr = (__real_t *) residual.request().ptr;
    auto *is_residual_calculated_ptr = (bool *)is_residual_calculated.request().ptr;
    auto *X_ptr = (__real_t *) X.request().ptr;
    auto *ns_x_mesh_phi_old = (__real_t *)((py::array_t<__real_t>)(graph.attr("ns_x_mesh").attr("phi_old"))).request().ptr;
    auto *ns_y_mesh_phi_old = (__real_t *)((py::array_t<__real_t>)(graph.attr("ns_y_mesh").attr("phi_old"))).request().ptr;

    // Initialization
    for(__integer_t i = 0; i < (__integer_t)residual_size; ++i) {
        is_residual_calculated_ptr[i] = false;
    }

    // Knowns (Constants)
    auto dt = (__real_t)py::float_(graph.attr("dt"));
    auto dx = (__real_t)py::float_(graph.attr("dx"));
    auto dy = (__real_t)py::float_(graph.attr("dy"));
    auto rho = (__real_t)py::float_(graph.attr("rho"));
    auto mi = (__real_t)py::float_(graph.attr("mi"));
    auto bc = (__real_t)py::float_(graph.attr("bc")); // Velocity at the top (U_top)

    std::vector<__real_t> P;
    std::vector<__real_t> U;
    std::vector<__real_t> V;
    for (__integer_t i = 0; i < (__integer_t)pressure_mesh_size; ++i) {
        P.push_back(X_ptr[3 * i]);

        if ((i + 1) % pressure_mesh_nx != 0) {
            U.push_back(X_ptr[3 * i + 1]);
        }

        if (i < (__integer_t)ns_y_mesh_size) {
            V.push_back(X_ptr[3 * i + 2]);
        }
    }

    int begin, end, step;
    #pragma omp parallel private(begin, end, step)
    {
        auto thread_id = omp_get_thread_num();
        auto n_threads = omp_get_num_threads();

        // Residual function for conservation of mass
        step = (int)pressure_mesh_size / n_threads;
        begin = step * thread_id;
        end = begin + step;        
        if (thread_id == n_threads - 1 && pressure_mesh_size % n_threads != 0) end += 1;
        for (__integer_t i = begin; i < end; ++i) {
            auto j = (__integer_t)i / pressure_mesh_nx;
    
            // Index conversion
            auto i_U_w = i - j - 1;
            auto i_U_e = i_U_w + 1;
            auto i_V_n = i;
            auto i_V_s = i_V_n - pressure_mesh_nx;
    
            // Knowns
            auto is_left_boundary = i % pressure_mesh_nx == 0;
            auto is_right_boundary = (i + 1) % pressure_mesh_nx == 0;
            auto is_bottom_boundary = j % pressure_mesh_ny == 0;
            auto is_top_boundary = (j + 1) % pressure_mesh_ny == 0;
    
            // Unknowns
            auto U_w = is_left_boundary ? 0.0 : U[i_U_w];
            auto U_e = is_right_boundary ? 0.0 : U[i_U_e];
            auto V_n = is_top_boundary ? 0.0 : V[i_V_n];
            auto V_s = is_bottom_boundary ? 0.0 : V[i_V_s];
    
            // Conservation of Mass
            auto ii = 3 * i;
            residual_ptr[ii] = (U_e * dy - U_w * dy) + (V_n * dx - V_s * dx);
            is_residual_calculated_ptr[ii] = true;
        }
    
        // Residual function for Navier Stokes (X)
        step = (int)ns_x_mesh_size / n_threads;
        begin = step * thread_id;
        end = begin + step;        
        if (thread_id == n_threads - 1 && ns_x_mesh_size % n_threads != 0) end += 1;
        for (__integer_t i = begin; i < end; ++i) {
            auto j = (__integer_t)i / ns_x_mesh_nx;
    
            // Index conversion
            auto i_U_P = i;
            auto i_U_W = i - 1;
            auto i_U_E = i + 1;
            auto i_U_N = i + ns_x_mesh_nx;
            auto i_U_S = i - ns_x_mesh_nx;
            auto i_P_w = i + ((__integer_t)i / ns_x_mesh_nx);
            auto i_P_e = i_P_w + 1;
            auto i_V_NW = i + j;
            auto i_V_NE = i_V_NW + 1;
            auto i_V_SW = i_V_NW - ns_y_mesh_nx;
            auto i_V_SE = i_V_NE - ns_y_mesh_nx;
    
            // Knowns
            auto U_P_old = ns_x_mesh_phi_old[i_U_P];
            auto is_left_boundary = i % ns_x_mesh_nx == 0;
            auto is_right_boundary = (i + 1) % ns_x_mesh_nx == 0;
            auto is_bottom_boundary = j % ns_x_mesh_ny == 0;
            auto is_top_boundary = (j + 1) % ns_x_mesh_ny == 0;
            
            // Unknowns
            auto U_P = U[i_U_P];
            auto U_W = is_left_boundary ? 0.0 : U[i_U_W];
            auto U_E = is_right_boundary ? 0.0 : U[i_U_E];
            auto U_N = is_top_boundary ? bc : U[i_U_N];
            auto U_S = is_bottom_boundary ? 0.0 : U[i_U_S];
            auto P_w = P[i_P_w];
            auto P_e = P[i_P_e];
            auto V_NE = is_top_boundary ? 0.0 : V[i_V_NE];
            auto V_NW = is_top_boundary ? 0.0 : V[i_V_NW];
            auto V_SE = is_bottom_boundary ? 0.0 : V[i_V_SE];
            auto V_SW = is_bottom_boundary ? 0.0 : V[i_V_SW];
            
            // Calculated (Interpolated and Secondary Variables)
            auto dU_e_dx = (U_E - U_P) / dx;
            auto dU_w_dx = (U_P - U_W) / dx;
            auto dU_n_dx = (U_N - U_P) / dy;
            auto dU_s_dx = (U_P - U_S) / dy;
            auto U_e = (U_E + U_P) / 2.0;
            auto U_w = (U_P + U_W) / 2.0;
            auto V_n = (V_NE + V_NW) / 2.0;
            auto V_s = (V_SE + V_SW) / 2.0;
            auto beta_U_e = U_e > 0.0 ? 0.5 : -0.5;
            auto beta_U_w = U_w > 0.0 ? 0.5 : -0.5;
            auto beta_V_n = V_n > 0.0 ? 0.5 : -0.5;
            auto beta_V_s = V_s > 0.0 ? 0.5 : -0.5;
    
            // Navier Stokes X
            auto transient_term = (rho * U_P - rho * U_P_old) * (dx * dy / dt);
            auto advective_term = \
                rho * U_e * ((.5 - beta_U_e) * U_E + (.5 + beta_U_e) * U_P) * dy - \
                rho * U_w * ((.5 - beta_U_w) * U_P + (.5 + beta_U_w) * U_W) * dy + \
                rho * V_n * ((.5 - beta_V_n) * U_N + (.5 + beta_V_n) * U_P) * dx - \
                rho * V_s * ((.5 - beta_V_s) * U_P + (.5 + beta_V_s) * U_S) * dx;
            auto difusive_term = \
                mi * dU_e_dx * dy - \
                mi * dU_w_dx * dy + \
                mi * dU_n_dx * dx - \
                mi * dU_s_dx * dx;
            auto source_term = -(P_e - P_w) * dy;
    
            auto ii = 3 * (i + j) + 1;
            residual_ptr[ii] = transient_term + advective_term - difusive_term - source_term;
            is_residual_calculated_ptr[ii] = true;
        }
    
        // Residual function for Navier Stokes (Y)
        step = (int)ns_y_mesh_size / n_threads;
        begin = step * thread_id;
        end = begin + step;        
        if (thread_id == n_threads - 1 && ns_y_mesh_size % n_threads != 0) end += 1;
        for (__integer_t i = begin; i < end; ++i) {
            auto j = (__integer_t)i / ns_y_mesh_nx;
    
            // Index conversion
            auto i_V_P = i;
            auto i_V_W = i - 1;
            auto i_V_E = i + 1;
            auto i_V_N = i + ns_y_mesh_nx;
            auto i_V_S = i - ns_y_mesh_nx;
            auto i_P_n = i + ns_y_mesh_nx;
            auto i_P_s = i;
            auto i_U_SE = i - j;
            auto i_U_SW = i_U_SE - 1;
            auto i_U_NE = i_U_SE + ns_x_mesh_nx;
            auto i_U_NW = i_U_SW + ns_x_mesh_nx;
    
            // Knowns
            auto V_P_old = ns_y_mesh_phi_old[i_V_P];
            auto is_left_boundary = i % ns_y_mesh_nx == 0;
            auto is_right_boundary = (i + 1) % ns_y_mesh_nx == 0;
            auto is_bottom_boundary = j % ns_y_mesh_ny == 0;
            auto is_top_boundary = (j + 1) % ns_y_mesh_ny == 0;
    
            // Unknowns
            auto V_P = V[i_V_P];
            auto V_E = is_right_boundary ? 0.0 : V[i_V_E];
            auto V_W = is_left_boundary ? 0.0 : V[i_V_W];
            auto V_N = is_top_boundary ? 0.0 : V[i_V_N];
            auto V_S = is_bottom_boundary ? 0.0 : V[i_V_S];
            auto P_n = P[i_P_n];
            auto P_s = P[i_P_s];
            auto U_SE = is_right_boundary ? 0.0 : U[i_U_SE];
            auto U_SW = is_left_boundary ? 0.0 : U[i_U_SW];
            auto U_NE = is_right_boundary ? 0.0 : (is_top_boundary ? bc : U[i_U_NE]);
            auto U_NW = is_left_boundary ? 0.0 : (is_top_boundary ? bc : U[i_U_NW]);
    
            // Calculated (Interpolated and Secondary Variables)
            auto dV_e_dx = (V_E - V_P) / dx;
            auto dV_w_dx = (V_P - V_W) / dx;
            auto dV_n_dx = (V_N - V_P) / dy;
            auto dV_s_dx = (V_P - V_S) / dy;
            auto U_e = (U_NE + U_SE) / 2.0;
            auto U_w = (U_NW + U_SW) / 2.0;
            auto V_n = (V_P + V_N) / 2.0;
            auto V_s = (V_S + V_P) / 2.0;
            auto beta_U_e = U_e > 0.0 ? 0.5 : -0.5;
            auto beta_U_w = U_w > 0.0 ? 0.5 : -0.5;
            auto beta_V_n = V_n > 0.0 ? 0.5 : -0.5;
            auto beta_V_s = V_s > 0.0 ? 0.5 : -0.5;
    
            // Navier Stokes Y
            auto transient_term = (rho * V_P - rho * V_P_old) * (dx * dy / dt);
            auto advective_term = \
                rho * U_e * ((.5 - beta_U_e) * V_E + (.5 + beta_U_e) * V_P) * dy - \
                rho * U_w * ((.5 - beta_U_w) * V_P + (.5 + beta_U_w) * V_W) * dy + \
                rho * V_n * ((.5 - beta_V_n) * V_N + (.5 + beta_V_n) * V_P) * dx - \
                rho * V_s * ((.5 - beta_V_s) * V_P + (.5 + beta_V_s) * V_S) * dx;
            auto difusive_term = \
                mi * dV_e_dx * dy - \
                mi * dV_w_dx * dy + \
                mi * dV_n_dx * dx - \
                mi * dV_s_dx * dx;
            auto source_term = -(P_n - P_s) * dx;
    
            auto ii = 3 * i + 2;
            residual_ptr[ii] = transient_term + advective_term - difusive_term - source_term;
            is_residual_calculated_ptr[ii] = true;
        }
    
        // Set all remaining residuals with x[i] - x[i] = R
        // Basically, will avoid None on equations for U_dummy that have no equation attached.
        //
        step = (int)residual_size / n_threads;
        begin = step * thread_id;
        end = begin + step;        
        if (thread_id == n_threads - 1 && residual_size % n_threads != 0) end += 1;
        for (__integer_t ii = begin; ii < end; ++ii) {
            if (!is_residual_calculated_ptr[ii]) {
                residual_ptr[ii] = X_ptr[ii];
            }
        }
    }

    return residual;
}
