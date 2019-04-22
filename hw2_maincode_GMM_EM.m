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
%% PART 2: GMM ////////////////////////////////////////////////////////////////////////



rand('seed',0)

%% Divide DATA TRAIN into two part to creat 2 GMM for each class
% labeled:Class 1

       class1_data_a= read_data_train(find(read_data_train(1:16018,2049) == 1),1:2048);
      [m1_hat_a, S1_hat_a]=Gaussian_ML_estimate(class1_data_a');
       p1_a= length(find(read_data_train(1:16018,2049)==1))/16018;
      class1_data_b= read_data_train(find(read_data_train(16019:32036,2049) == 1),1:2048);
      [m1_hat_b, S1_hat_b]=Gaussian_ML_estimate(class1_data_b');
       p1_b= length(find(read_data_train(16019:32036,2049)==1))/16018;
       
    
% labeled:Class 2
      class2_data_a= read_data_train(find(read_data_train(1:16018,2049) == 2),1:2048);
      [m2_hat_a, S2_hat_a]=Gaussian_ML_estimate(class2_data_a');
       p2_a= length(find(read_data_train(1:16018,2049)==2))/16018;
      class2_data_b= read_data_train(find(read_data_train(16019:32036,2049) == 2),1:2048);
      [m2_hat_b, S2_hat_b]=Gaussian_ML_estimate(class2_data_b');
       p2_b= length(find(read_data_train(16019:32036,2049)==2))/16018;
        
% labeled:Class 3
       class3_data_a= read_data_train(find(read_data_train(1:16018,2049) == 3),1:2048);
      [m3_hat_a, S3_hat_a]=Gaussian_ML_estimate(class3_data_a');
       p3_a= length(find(read_data_train(1:16018,2049)==3))/16018;
      class3_data_b= read_data_train(find(read_data_train(16019:32036,2049) == 3),1:2048);
      [m3_hat_b, S3_hat_b]=Gaussian_ML_estimate(class3_data_b');
       p3_b= length(find(read_data_train(16019:32036,2049)==3))/16018;  
% labeled:Class 4
       
         class4_data_a= read_data_train(find(read_data_train(1:16018,2049) == 4),1:2048);
      [m4_hat_a, S4_hat_a]=Gaussian_ML_estimate(class4_data_a');
       p4_a= length(find(read_data_train(4:16018,2049)==4))/16018;
      class4_data_b= read_data_train(find(read_data_train(16019:32036,2049) == 4),1:2048);
      [m4_hat_b, S4_hat_b]=Gaussian_ML_estimate(class4_data_b');
       p4_b= length(find(read_data_train(16019:32036,2049)==4))/16018;
% labeled:Class 5
     class5_data_a= read_data_train(find(read_data_train(1:16018,2049) == 5),1:2048);
      [m5_hat_a, S5_hat_a]=Gaussian_ML_estimate(class5_data_a');
       p5_a= length(find(read_data_train(1:16018,2049)==5))/16018;
      class5_data_b= read_data_train(find(read_data_train(16019:32036,2049) == 5),1:2048);
      [m5_hat_b, S5_hat_b]=Gaussian_ML_estimate(class5_data_b');
       p5_b= length(find(read_data_train(16019:32036,2049)==5))/16018;
      
% labeled:Class 6
        class6_data_a= read_data_train(find(read_data_train(1:16018,2049) == 6),1:2048);
      [m6_hat_a, S6_hat_a]=Gaussian_ML_estimate(class6_data_a');
       p6_a= length(find(read_data_train(1:16018,2049)==6))/16018;
      class6_data_b= read_data_train(find(read_data_train(16019:32036,2049) == 6),1:2048);
      [m6_hat_b, S6_hat_b]=Gaussian_ML_estimate(class6_data_b');
       p6_b= length(find(read_data_train(16019:32036,2049)==6))/16018;
% labeled:Class 7
          class7_data_a= read_data_train(find(read_data_train(1:16018,2049) == 7),1:2048);
      [m7_hat_a, S7_hat_a]=Gaussian_ML_estimate(class7_data_a');
       p7_a= length(find(read_data_train(1:16018,2049)==7))/16018;
      class7_data_b= read_data_train(find(read_data_train(16019:32036,2049) == 7),1:2048);
      [m7_hat_b, S7_hat_b]=Gaussian_ML_estimate(class7_data_b');
       p7_b= length(find(read_data_train(16019:32036,2049)==7))/16018;
% labeled:Class 8
       class8_data_a= read_data_train(find(read_data_train(1:16018,2049) == 8),1:2048);
      [m8_hat_a, S8_hat_a]=Gaussian_ML_estimate(class8_data_a');
       p8_a= length(find(read_data_train(1:16018,2049)==8))/16018;
      class8_data_b= read_data_train(find(read_data_train(16019:32036,2049) == 8),1:2048);
      [m8_hat_b, S8_hat_b]=Gaussian_ML_estimate(class8_data_b');
       p8_b= length(find(read_data_train(16019:32036,2049)==8))/16018;
% labeled:Class 9
      class9_data_a= read_data_train(find(read_data_train(1:16018,2049) == 9),1:2048);
      [m9_hat_a, S9_hat_a]=Gaussian_ML_estimate(class9_data_a');
       p9_a= length(find(read_data_train(1:16018,2049)==9))/16018;
      class9_data_b= read_data_train(find(read_data_train(16019:32036,2049) == 9),1:2048);
      [m9_hat_b, S9_hat_b]=Gaussian_ML_estimate(class9_data_b');
       p9_b= length(find(read_data_train(16019:32036,2049)==9))/16018;
 % labeled:Class 10
      class10_data_a= read_data_train(find(read_data_train(1:16018,2049) == 10),1:2048);
      [m10_hat_a, S10_hat_a]=Gaussian_ML_estimate(class10_data_a');
       p10_a= length(find(read_data_train(1:16018,2049)==10))/16018;
      class10_data_b= read_data_train(find(read_data_train(16019:32036,2049) == 10),1:2048);
      [m10_hat_b, S10_hat_b]=Gaussian_ML_estimate(class10_data_b');
       p10_b= length(find(read_data_train(16019:32036,2049)==10))/16018;
 %% To estimate the Gaussian mixture model of each class

 train=read_data_train(:,1:2048)';
 train_label= read_data_train(:,2049)';
 test=read_data_test(:,1:2048)';
 test_label= read_data_test(:,2049)';


% 1. Estimate the Gaussian mixture model of each class
% class 1
m11_ini=m1_hat_a; m12_ini=m1_hat_b; 
m1_ini=[m11_ini m12_ini];
S1_ini=[mean(S1_hat_a(:,1)) mean(S1_hat_b(:,1))];
w1_ini=[p1_a p1_b];
% class 2
m21_ini=m2_hat_a; m22_ini=m2_hat_b; 
m2_ini=[m21_ini m22_ini];
S2_ini=[mean(S2_hat_a(:,1)) mean(S2_hat_b(:,1))];;
w2_ini=[p2_a p2_b];
% class 3
m31_ini=m3_hat_a; m32_ini=m3_hat_b; 
m3_ini=[m31_ini m32_ini];
S3_ini=[mean(S3_hat_a(:,1)) mean(S3_hat_b(:,1))];
w3_ini=[p3_a p3_b];
% class 4
m41_ini=m4_hat_a; m42_ini=m4_hat_b; 
m4_ini=[m41_ini m42_ini];
S4_ini=[mean(S4_hat_a(:,1)) mean(S4_hat_b(:,1))];
w4_ini=[p4_a p4_b];
% class 5
m51_ini=m5_hat_a; m52_ini=m5_hat_b; 
m5_ini=[m51_ini m52_ini];
S5_ini=[mean(S5_hat_a(:,1)) mean(S5_hat_b(:,1))];
w5_ini=[p4_a p4_b];
% class 6
m61_ini=m6_hat_a; m62_ini=m6_hat_b; 
m6_ini=[m61_ini m62_ini];
S6_ini=[mean(S6_hat_a(:,1)) mean(S6_hat_b(:,1))];
w6_ini=[p6_a p6_b];
% class 7
m71_ini=m7_hat_a; m72_ini=m7_hat_b; 
m7_ini=[m71_ini m72_ini];
S7_ini=[mean(S7_hat_a(:,1)) mean(S7_hat_b(:,1))];
w7_ini=[p7_a p7_b];
% class 8
m81_ini=m8_hat_a; m82_ini=m8_hat_b; 
m8_ini=[m81_ini m82_ini];
S8_ini=[mean(S8_hat_a(:,1)) mean(S8_hat_b(:,1))];
w8_ini=[p8_a p8_b];
% class 9
m91_ini=m9_hat_a; m92_ini=m9_hat_b; 
m9_ini=[m91_ini m92_ini];
S9_ini=[mean(S9_hat_a(:,1)) mean(S9_hat_b(:,1))];
w9_ini=[p9_a p9_b];
% class 10
m101_ini=m10_hat_a; m102_ini=m10_hat_b; 
m10_ini=[m101_ini m102_ini];
S10_ini=[mean(S10_hat_a(:,1)) mean(S10_hat_b(:,1))];
w10_ini=[p10_a p10_b];


m_ini{1}=m1_ini;m_ini{2}=m2_ini;m_ini{3}=m3_ini;m_ini{4}=m4_ini;m_ini{5}=m5_ini;m_ini{6}=m6_ini;m_ini{7}=m7_ini;m_ini{8}=m8_ini;m_ini{9}=m9_ini;m_ini{10}=m10_ini;
S_ini{1}=S1_ini;S_ini{2}=S2_ini;S_ini{3}=S3_ini;S_ini{4}=S4_ini;S_ini{5}=S5_ini;S_ini{6}=S6_ini;S_ini{7}=S7_ini;S_ini{8}=S8_ini;S_ini{9}=S9_ini;S_ini{10}=S10_ini;
w_ini{1}=w1_ini;w_ini{2}=w2_ini;w_ini{3}=w3_ini;w_ini{4}=w4_ini;w_ini{5}=w5_ini;w_ini{6}=w6_ini;w_ini{7}=w7_ini;w_ini{8}=w8_ini;w_ini{9}=w9_ini;w_ini{10}=w10_ini;
[m_hat,S_hat,w_hat,P_hat]=EM_pdf_est(train,train_label,m_ini,S_ini,w_ini);
m1=p1_a*m_hat{1}(:,1) + p1_b*m_hat{1}(:,2);m2= p2_a*m_hat{2}(:,1) + p2_b*m_hat{2}(:,2);m3=p3_a*m_hat{3}(:,1) + p3_b*m_hat{3}(:,2);m4=p4_a*m_hat{4}(:,1) + p4_b*m_hat{4}(:,2);m5=p5_a*m_hat{5}(:,1) + p5_b*m_hat{5}(:,2);
m6=p6_a*m_hat{6}(:,1) + p6_b*m_hat{6}(:,2);m7=p7_a*m_hat{7}(:,1) + p7_b*m_hat{7}(:,2);m8=p8_a*m_hat{8}(:,1) + p8_b*m_hat{8}(:,2);m9=p9_a *m_hat{9}(:,1) + p9_b*m_hat{9}(:,2);m10=p10_a*(m_hat{10}(:,1) + p10_b*m_hat{10}(:,2));
m=[m1 m2 m3 m4 m5 m6 m7 m8 m9 m10];

% length=length(S_hat{10})
test=read_data_test(:,1:2048);
% Euclidean distance classifier
z_euclidean=euclidean_classifier(m,test');

%  Mahalanobis distance classifier
%z_mahalanobis=mahalanobis_classifier(m_hat,S_hat,test');

%  bayes classifier and provide as input the matrices

%z_bayesian=bayes_classifier(m_hat,S,p,test');
%% Error calculation

err_euclidean = (1-length(find(read_data_test(:,2049)==z_euclidean'))/8009)
%err_mahalanobis = (1-length(find(read_data_test(:,2049)==z_mahalanobis'))/8009)
%err_bayesian = (1-length(find(read_data_test(:,2049)==z_bayesian'))/8009)

% 2. Use function mixture_Bayes to classify the data vectors of Z and function compute_error to
% obtain the classification error.
%for j=1:10
   % le=length(S_hat{j});
    %te=[];
    %for i=1:le
        %te(:,:,i)=S_hat{j}(i)*eye(2048);
    %end
    %S{j}=te;
%end
%[y_est]=mixture_Bayes(m_hat,S,w_hat,P_hat,test);
%[classification_error]=compute_error(test_label,y_est)

