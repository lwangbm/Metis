function [all,ratio,abs_vio] = testbed_more(Num_C_per_app, nodes)
Num_apps = 7;
Dn = 1000;
Num_nodes = nodes;
Capacity = 8;
preference = table2array(readtable("./interference.csv"));

constraint_set = [];
[N_C,tmp] = size(preference);
for i = 1: N_C
    weight = 1;
    if ((preference(i,1)==1) && (preference(i,2) ==2)) ||((preference(i,1)==1) && (preference(i,2) ==7))
        weight = 2;
    end
    constraint_set = [constraint_set; [preference(i,1),preference(i,2),preference(i,3),preference(i,4),weight]]; 
end

%% parameter
total_C = sum(Num_C_per_app);
total_variavle_X = total_C * Num_nodes;
[N_C,tmp] = size(constraint_set);
app_s = constraint_set(:,1);
app_d = Num_C_per_app(1,app_s);
[a0,b0] = histc(app_s,unique(app_s));
c0 = zeros(Num_apps,1);
c0(unique(app_s),1) = a0;
N_variables = total_variavle_X + sum(c0' .* Num_C_per_app) * Num_nodes;
w1 = 0;
w2 = 1;
f = -[w1 * ones(1,total_variavle_X), - w2 * ones(1,sum(c0' .* Num_C_per_app) * Num_nodes)];
ic = [1:1:N_variables];
endd = 1;


core_constraint = 0;
for ct = 1: N_C
    constraint = constraint_set(ct,:);
    core_constraint = core_constraint + Num_C_per_app(1,constraint(1));

end
rows_length = Num_nodes + sum(Num_C_per_app)*2 + Num_nodes*core_constraint;
cols_length = N_variables;
A = sparse(rows_length,cols_length);
b = [];

for node = 1:Num_nodes
    node_start = (node - 1) * sum(Num_C_per_app);
    c_tag_start = node_start + 1;
    c_tag_stop = node_start + sum(Num_C_per_app);
    a = zeros(1,N_variables);
    a(c_tag_start : 1 : c_tag_stop) = 1;
    b_2 = Capacity;
    A(endd,:) = a; endd=endd+1;
    b = [b;b_2];
end

for app = 1:Num_apps
    for con_index = 1:Num_C_per_app(1,app)
        a = zeros(1,N_variables);
        for node = 1:Num_nodes

            a((node - 1) * sum(Num_C_per_app) + sum(Num_C_per_app(1:app-1)) + con_index)  = 1;


        end
        b_2 = 1;
        A(endd,:) = a; endd=endd+1;
        A(endd,:) = -a; endd=endd+1;
        b = [b;b_2];b = [b;-b_2];
    end
end

app_s = app_s';

AAA= A(1:endd-1,:);

N_C_0 = floor(N_C/10);
N_C_1 = floor(N_C - 9*N_C_0);

for index=0:8
endd = 1;
A = sparse(rows_length,cols_length);
for ct = 1 + index * N_C_0 : (index+1) * N_C_0
    constraint = constraint_set(ct,:);
    constraint_index = ct;
    max_or_min = constraint(3);
    assert (max_or_min <=1)
    if max_or_min == 1
        a = zeros(1,N_variables);
    
        for node = 1:Num_nodes
            node_start = (node - 1) * sum(Num_C_per_app);
            c_tag_start = node_start + sum(Num_C_per_app(1,1:(constraint(2)-1))) + 1;
            c_tag_stop = node_start + sum(Num_C_per_app(1,1:(constraint(2)))) ;
            a_1 = a;
            a_1(c_tag_start : 1 : c_tag_stop) = -1;

            for container_index = 1:Num_C_per_app(1,constraint(1))
                constraint_id = sum(app_d(1,1:(constraint_index-1))) * Num_nodes + (node - 1) * Num_C_per_app(1,constraint(1)) + container_index + total_variavle_X;
                s_tag_id = (node - 1) * sum(Num_C_per_app) + sum(Num_C_per_app(1,1:(constraint(1)-1))) + container_index;
                a_2 = a_1;
                a_2(s_tag_id) = Dn;
                a_2(constraint_id) = -1;
                b_2 = Dn - constraint(4);
                A(endd,:) = a_2; endd=endd+1;
                b = [b;b_2];
                if constraint(5)>1
                    test =1;
                end
                f(constraint_id) = f(constraint_id)*constraint(5);

            end

        end
    else
        a = zeros(1,N_variables);
    
        for node = 1:Num_nodes
            node_start = (node - 1) * sum(Num_C_per_app);
            c_tag_start = node_start + sum(Num_C_per_app(1,1:(constraint(2)-1))) + 1;
            c_tag_stop = node_start + sum(Num_C_per_app(1,1:(constraint(2)))) ;
            a_1 = a;
            a_1(c_tag_start : 1 : c_tag_stop) = 1;

            for container_index = 1:Num_C_per_app(1,constraint(1))
                constraint_id = sum(app_d(1,1:(constraint_index-1))) * Num_nodes + (node - 1) * Num_C_per_app(1,constraint(1)) + container_index + total_variavle_X;
                s_tag_id = (node - 1) * sum(Num_C_per_app) + sum(Num_C_per_app(1,1:(constraint(1)-1))) + container_index;
                a_2 = a_1;
                a_2(s_tag_id) = Dn;
                a_2(constraint_id) = -1;
                b_2 = Dn + constraint(4);
                A(endd,:) = a_2; endd=endd+1;
                b = [b;b_2];
                if constraint(5)>1
                    test =1;
                end
                f(constraint_id) = f(constraint_id)*constraint(5);

            end

        end
    end
end

A=sparse(A);
A = A(1:endd-1,:);
AAA= [AAA;A];
AAA=sparse(AAA);
    
end

endd = 1;
A = sparse(rows_length,cols_length);
for ct = 1 + 9 * N_C_0 : N_C
    constraint = constraint_set(ct,:);
    constraint_index = ct;
    max_or_min = constraint(3);
    assert (max_or_min <=1)
    if max_or_min == 1
        a = zeros(1,N_variables);
    
        for node = 1:Num_nodes
            node_start = (node - 1) * sum(Num_C_per_app);
            c_tag_start = node_start + sum(Num_C_per_app(1,1:(constraint(2)-1))) + 1;
            c_tag_stop = node_start + sum(Num_C_per_app(1,1:(constraint(2)))) ;
            a_1 = a;
            a_1(c_tag_start : 1 : c_tag_stop) = -1;

            for container_index = 1:Num_C_per_app(1,constraint(1))
                constraint_id = sum(app_d(1,1:(constraint_index-1))) * Num_nodes + (node - 1) * Num_C_per_app(1,constraint(1)) + container_index + total_variavle_X;
                s_tag_id = (node - 1) * sum(Num_C_per_app) + sum(Num_C_per_app(1,1:(constraint(1)-1))) + container_index;
                a_2 = a_1;
                a_2(s_tag_id) = Dn;
                a_2(constraint_id) = -1;
                b_2 = Dn - constraint(4);
                A(endd,:) = a_2; endd=endd+1;
                b = [b;b_2];
                if constraint(5)>1
                    test =1;
                end
                f(constraint_id) = f(constraint_id)*constraint(5);

            end

        end
    else
        a = zeros(1,N_variables);
    
        for node = 1:Num_nodes
            node_start = (node - 1) * sum(Num_C_per_app);
            c_tag_start = node_start + sum(Num_C_per_app(1,1:(constraint(2)-1))) + 1;
            c_tag_stop = node_start + sum(Num_C_per_app(1,1:(constraint(2)))) ;
            a_1 = a;
            a_1(c_tag_start : 1 : c_tag_stop) = 1;

            for container_index = 1:Num_C_per_app(1,constraint(1))
                constraint_id = sum(app_d(1,1:(constraint_index-1))) * Num_nodes + (node - 1) * Num_C_per_app(1,constraint(1)) + container_index + total_variavle_X;
                s_tag_id = (node - 1) * sum(Num_C_per_app) + sum(Num_C_per_app(1,1:(constraint(1)-1))) + container_index;
                a_2 = a_1;
                a_2(s_tag_id) = Dn;
                a_2(constraint_id) = -1;
                b_2 = Dn + constraint(4);
                A(endd,:) = a_2; endd=endd+1;
                b = [b;b_2];
                if constraint(5)>1
                    test =1;
                end
                f(constraint_id) = f(constraint_id)*constraint(5);

            end

        end
    end
end

A=sparse(A);
A = A(1:endd-1,:);
AAA= [AAA;A];
AAA=sparse(AAA);

A = AAA;

%% opt
lb_12=zeros(N_variables,1);
ub_12=[ones(total_variavle_X,1);1e+10*ones(N_variables-total_variavle_X,1)];
options = optimoptions('intlinprog','MaxTime',60*60*3,'MaxFeasiblePoints', 6);

[x_12,fval_12,flag_12, output]=intlinprog(f,ic,A,b,[],[],lb_12,[],options);
% results
output
allocation = zeros(Num_nodes,Num_apps);
x_12(1:total_variavle_X);
index_x = 1;
for node_id = 1:Num_nodes
    for app_id = 1:Num_apps
        start_point = (node_id-1) * sum(Num_C_per_app) + sum(Num_C_per_app(1,1:(app_id-1)))  + 1;
        stop_point = (node_id-1) * sum(Num_C_per_app) + sum(Num_C_per_app(1,1:(app_id)));
        allocation(node_id,app_id) = sum(x_12(start_point:1:stop_point));
    end
end
allocation = uint8(allocation)
sum(allocation,1)
%
% print_array(allocation,Num_nodes,Num_apps)
f1 = -[w1 * ones(1,total_variavle_X), - w2 * ones(1,sum(c0' .* Num_C_per_app) * Num_nodes)];
% f1*x_12
% sum(c0' .* Num_C_per_app) * Num_nodes
ratio = f*x_12 / (sum(c0' .* Num_C_per_app) * Num_nodes);
all = reshape(allocation,[1,Num_nodes*Num_apps]);
abs_vio = f*x_12;
end

