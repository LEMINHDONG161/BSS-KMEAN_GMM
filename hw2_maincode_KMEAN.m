close all;
clear all
%% Load Data File
load('C:\Users\jys\Desktop\Artificial Inteligence\AI\new_data10'); 
read_data_train = data_train;
read_data_test = data_test;
% Result
%read_data_train 32036 x2049 ; 
%read_data_test  8009x2049;
% 2049 column is the class number of 1 to 10
%% PART 1: Clustering ////////////////////////////////////////////////////////////////////////
%% K mean input

m=10;
train=read_data_train(:,1:2048);
test=read_data_test(:,1:2048);
[l,N]=size(test');
rand('seed',0)
%% Choose initial vector by random selecting
%theta_ini=rand(l,m);
%% Choose initial vector by calculate the mean value of data_train
% labeled:Class 1

       class1_data= read_data_train(find(read_data_train(:,2049) == 1),1:2048);
      [m1_hat, S1_hat]=Gaussian_ML_estimate(class1_data');
    p1= length(find(read_data_train(:,2049)==1))/32036;

% labeled:Class 2
        class2_data= read_data_train(find(read_data_train(:,2049) == 2),1:2048);
         [m2_hat, S2_hat]=Gaussian_ML_estimate(class2_data');
          p2= length(find(read_data_train(:,2049)==2))/32036;
% labeled:Class 3
        class3_data= read_data_train(find(read_data_train(:,2049) == 3),1:2048);
         [m3_hat, S3_hat]=Gaussian_ML_estimate(class3_data');
          p3= length(find(read_data_train(:,2049)==3))/32036;
% labeled:Class 4
       class4_data= read_data_train(find(read_data_train(:,2049) == 4),1:2048);
       [m4_hat, S4_hat]=Gaussian_ML_estimate(class4_data');
        p4= length(find(read_data_train(:,2049)==4))/32036;
% labeled:Class 5
        class5_data= read_data_train(find(read_data_train(:,2049) == 5),1:2048);
       [m5_hat, S5_hat]=Gaussian_ML_estimate(class5_data');
        p5= length(find(read_data_train(:,2049)==5))/32036;
% labeled:Class 6
       class6_data= read_data_train(find(read_data_train(:,2049) == 6),1:2048);
       [m6_hat, S6_hat]=Gaussian_ML_estimate(class6_data');
        p6= length(find(read_data_train(:,2049)==6))/32036;
% labeled:Class 7
       class7_data= read_data_train(find(read_data_train(:,2049) == 7),1:2048);
         [m7_hat, S7_hat]=Gaussian_ML_estimate(class7_data');
         p7= length(find(read_data_train(:,2049)==7))/32036;
% labeled:Class 8
      class8_data= read_data_train(find(read_data_train(:,2049) == 8),1:2048);
        [m8_hat, S8_hat]=Gaussian_ML_estimate(class8_data');
         p8= length(find(read_data_train(:,2049)==8))/32036;
% labeled:Class 9
     class9_data= read_data_train(find(read_data_train(:,2049) == 9),1:2048);
     [m9_hat, S9_hat]=Gaussian_ML_estimate(class9_data');
      p9= length(find(read_data_train(:,2049)==9))/32036;
% labeled:Class 10
     class10_data= read_data_train(find(read_data_train(:,2049) == 10),1:2048);
     [m10_hat, S10_hat]=Gaussian_ML_estimate(class10_data');
      p10= length(find(read_data_train(:,2049)==10))/32036;
   
 %m_hat and S_hat and p
   m_hat_train=[m1_hat m2_hat m3_hat m4_hat m5_hat m6_hat m7_hat m8_hat m9_hat m10_hat];
  % S_hat=(1/10)*(S1_hat+S2_hat+S3_hat+S4_hat+S5_hat+S6_hat+S7_hat+S8_hat+S9_hat+S10_hat);
 
   p=[p1 p2 p3 p4 p5 p6 p7 p8 p9 p10]';
 %p=[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1];
   S(:,:,1)=S1_hat;  S(:,:,2)=S2_hat; S(:,:,3)=S3_hat; S(:,:,4)=S4_hat; S(:,:,5)=S5_hat; 
   S(:,:,6)=S6_hat;  S(:,:,7)=S7_hat; S(:,:,8)=S8_hat; S(:,:,9)=S9_hat; S(:,:,10)=S10_hat; 
  %% Chose the initial by calculate the average value in test data
 
  sum_train =zeros(2048,1);
  for i=1:1:32036
      sum_train=sum_train+train(i,1:2048)';
  end
  mean_train=sum_train./32036;
  
  sum =zeros(2048,1);
  for i=1:1:8009
      sum=sum+test(i,1:2048)';
  end
  %mean_test=sum./8009;
  var_mean= mean_test - mean_train;
  
  mean_test1 = m1_hat + var_mean*p1; mean_test2 = m2_hat + var_mean*p2; mean_test3 = m3_hat + var_mean*p3; mean_test4 = m4_hat + var_mean*p4; mean_test5 = m5_hat + var_mean*p5; 
  mean_test6 = m6_hat + var_mean*p6; mean_test7 = m7_hat + var_mean*p7; mean_test8 = m8_hat + var_mean*p8; mean_test9 = m9_hat + var_mean*p9; mean_test10 = m10_hat + var_mean*p10; 
 
  m_hat_test=[mean_test1 mean_test2 mean_test3 mean_test4 mean_test5 mean_test6 mean_test7 mean_test8 mean_test9 mean_test10];
  %m_hat_test=0.5 * (m_hat_train + m_hat_test);
  
  %% K mean function: stopping when the update representative vector is the same with previous version
%[theta,bel,J, iter]=k_means(test',theta_ini);
%[theta_train,bel_train,J_train, iter_train]=k_means(test',m_hat_train);
[theta_test,bel_test,J_test, iter_test]=k_means(test',m_hat_test);
%% K mean function: limitation of sum of distance quit because it confuse with K mean algorithm
%[theta_train_sum,bel_train_sum,J_train_sum, iter_train_sum]=k_means_sum(test',m_hat);
%% Error calculation
%err_Kmean_random= (1-length(find(read_data_test(:,2049)==bel'))/8009)
%err_Kmean_train = (1-length(find(read_data_test(:,2049)==bel_train'))/8009)
err_Kmean_test = (1-length(find(read_data_test(:,2049)==bel_test'))/8009)

