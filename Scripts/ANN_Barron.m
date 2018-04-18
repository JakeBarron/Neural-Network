%ANN with 1 hidden layer with randomly generated number of nodes
%originally published by T. Hoque
%modified by Jake Barron

% This set of ANNs will classify flowers based on measured traits of the flower
%/////////////////////////////////////////////////meta setup//////////////////////////////////////////
%defining 5 ANNs from prefabricated variables with random sized hidden layers
load('ANNs.mat', 'L1', 'L2','L3','L4','L5')
L1
L2
L3
L4
L5
% ///////////////////////////////////////////////// Training Section ////////////////////////////////////////////////
% ////////// initialization
% // Defining the layers: 4 input features, 3 output classes
for ANN = 1: 5
  if ANN == 1
    L = L1;
  elseif ANN == 2
    L = L2;
  elseif ANN == 3
    L = L3;
  elseif ANN == 4
    L = L4;
  elseif ANN == 5
    L = L5;
  end
  alpha = 0.3;   % //usually alpha < 0, ranging from 0.1 to 1
  target_mse=0.05; % // one of the exit condition
  Max_Epoch=200;  % // one of the exit condition
  Min_Error=Inf;
  Min_Error_Epoch=-1;
  epoch=0;       % // 1 epoch => One forward and backward sweep of the net for each training sample 
  mse =Inf;      % // initializing the Mean Squared Error with a very large value.
  Err=[];
  Epo=[];

  % ////////// load datasets
  load X.txt      % // contains features: Column1: x1 (sepal length) and Column2: x2 (sepal width) ...
                  % // Column3: x3 (Petal length) Column4: x4 (Petal width)
  [Nx,P]=size(X); % // Nx = # of sample in X, P= # of feature in X
  load Y.txt      % // Target Output
  [Ny,K]= deal(150,3); % // Ny = # of target output in Y, K= # of class for K classes when K>=3 otherwise, K=1 (for Binary case)

  % Optional: Since input and output are kept in different files, it is better to verify the loaded sample size/dimensions.
  if Nx ~= Ny 
        error ('The input/output sample sizes do not match');
  end

  % Optional
  if L(1) ~= P
         error ('The number of input nodes must be equal to the size of the features')' 
  end 

  % Optional
  if L(end) ~= K
         error ('The number of output nodes should be equal to K')' 
  end 
    
%//////////////////////////////////////////10 fold cross validation data setup///////////////////////
%outer loop for 10-fold cross validation

  %Let us allocate places for Term, T 
  T=cell(length(L),1);
  for i=1:length(L)
    T{i} =ones (L(i),1);
  end

  %Let us allocate places for activation, i.e., Z
  Z=cell(length(L),1);
  for i=1:length(L)-1
    Z{i} =zeros (L(i)+1,1); % it does not matter how do we initialize (with '0' or '1', or whatever,) this is fine!
  end
  Z{end} =zeros (L(end),1);  % at the final layer there is no Bias unit

  %Let us allocate places for error term delta, d
  d=cell(length(L),1);
  dSum=cell(length(L),1);
  for i=1:length(L)
    d{i} =zeros (L(i),1);
    dSum{i} =zeros (L(i),1);
  end

%validation index counter
  for k = 1:10
    Dtest = [];
    Dtrain = [];
    Ytest = [];
    Ytrain = [];
    for v = 1 : Nx
      if mod(v,10) == k-1 
        Dtest = [Dtest;X(v,:)];
        Ytest = [Ytest;Y(v,:)];
      else
        Dtrain = [Dtrain;X(v,:)];
        Ytrain = [Ytrain;Y(v,:)];
      end
    end
  
    [Ntrain, Ptrain] = size(Dtrain);
        
    B=cell(length(L)-1,1);  % forming the number of Beta/weight matrix needed in between the layers

    for i=1:length(L)-1        % Assign uniform random values in [-0.7, 0.7] 
          B{i} =[1.4.*rand(L(i)+1,L(i+1))-0.7];
    end 
    
    while (mse > target_mse) && (epoch < Max_Epoch)   % outer loop with exit conditions
      
      CSqErr=0; 		% //Cumulative Sq Err of each Sample; we will take the average after computing Nx_th sample (=> mse)
      deltaB=cell(length(L)-1,1);  % creating deltaB to hol

      for i=1:length(L)-1       
            deltaB{i} =[zeros(L(i)+1,L(i+1))];
      end 

      for j=1:Ntrain 		    % // for loop #1		
          Z{1} = [Dtrain(j,:) 1]';   % // Load Inputs with bias=1
          % // Load Corresponding Desired or Target output
          if Ytrain(j,1)' == 1
            Yk = [1,0,0]';
          elseif Ytrain(j,1)' == 2
            Yk = [0,1,0]';
          else
            Yk = [0,0,1]';
          end
          % forward propagation 
          % ----------------------
          for i=1:length(L)-1
            T{i+1} = B{i}' * Z{i};
                
            if (i+1)<length(L)
              Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
            else  
              Z{i+1}=(1./(1+exp(-T{i+1}))); 
            end 
          end  % // end of forward propagation 
             
          CSqErr= CSqErr+sum((Yk-Z{end}).^2);  % // collect sample wise Cumulative Sq Err

         % // Compute error term delta 'd' for each of the node except the input unit
         % -----------------------------------------------------------------------
         d{end}=(Z{end}-Yk).*Z{end}.*(1-Z{end}); % // delta error term for the output layer
         %dSum{end}=(Z{end}-Yk).*Z{end}.*(1-Z{end});
        
           for i=length(L)-1:-1:2 
              d{i}=Z{i}(1:end-1).*(1-Z{i}(1:end-1)).*sum(B{i}(1:end-1,:)*d{i+1}); % //compute the error term for all the hidden layer (and skip the input layer).
           end      
        
          for i=1:length(L)-1 
            deltaB{i}(1:end-1,:)=deltaB{i}(1:end-1,:)-alpha.*(Z{i}(1:end-1)*d{i+1}'); 
            deltaB{i}(end,:)=deltaB{i}(end,:)-alpha.*d{i+1}';  			% // update weight connected to the bias unit(or, intercept)	
           end     
      end  % //end of for loop #1
           %update the parameters/weights using batch learning
           %(Z{i}(1:end-1)*d{i+1}')
            for i=1:length(L)-1
              B{i} = B{i} + deltaB{i};
            end
        
        CSqErr= (CSqErr) /(Ntrain*3);        % //Average error of N training sample after an epoch 
        mse=CSqErr 
        epoch  = epoch+1
        
        Err = [Err mse];
        Epo = [Epo epoch];   


        if mse < Min_Error
            Min_Error=mse
            Min_Error_Epoch=epoch
            Bmin = B;
            
        end  
                    
    end % //while_end

          Min_Error
          Min_Error_Epoch  

    %//=============================================================================================================================
    %//The NN Node and structure needs to be saved, i.e. save L
        L
      
    %// Now the predicted weight B with least error should be saved in a file to be loaded and to be used for test set/new prediction

      %for i=max(size(B))
       %   B{i};
      %end 


    %// ================================================================================================================
    % ///////////////////////////////////////////////// Test Section ////////////////////////////////////////////////
      
    %// ====== Same (or similar) code as we used before for feed-forward part (see above)
      [Ntest, Ptest] = size(Dtest);
      TestCSqErr=0; 	
      for j=1:Ntest	    % for loop #1		
          
          Z{1} = [Dtest(j,:) 1]';  % Load Inputs with bias=1
          %%% //(Note: desired output here) .....  Yk   = Y(j,:)'; 	  % Load Corresponding Desired or Target output
          if Y(j,1)' == 1
            Yk = [1,0,0]';
          elseif Y(j,1)' == 2
            Yk = [0,1,0]';
          else
            Yk = [0,0,1]';
          end
          % // forward propagation 
          % //----------------------
          for i=1:length(L)-1
              T{i+1} = Bmin{i}' * Z{i};
             
              if (i+1)<length(L)
                Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
              else  
                Z{i+1}=(1./(1+exp(-T{i+1}))); 
              end 
          end  % //end of forward propagation 
           Z{end}; 
           
           TestCSqErr= TestCSqErr+sum((Yk-Z{end}).^2);  % // collect sample wise Cumulative Sq Err
           TestCSqErr= (TestCSqErr) /(Ntrain*3);      % //Average error of N sample after an epoch 
      end
  end
end   
  %===============================================================================================================
  %plot epoch versus error graph
 % plot (Epo,Err)  % plot based on full epoch

 % plot (Epo(1:200),CSqErr(1:200)) 
  

%TODO: categorization error - 3 probabilities what's prediction does it match training and test
%TODO: regularizationg