function generateIsomapLowDimTrainDataFun(directories)

 
    %% 0. Specify a data structure containing all the raw x data you want to 
    %% explicitly project using Isomap (for your training set data and images only). 

    Nsamps = 500; %number of samples per image 

    %%create regular grid of patches to extract from each training image -- could replace with randomized?
    xind_R = floor(linspace(1,3072,Nsamps)); 
    xind_Q = setdiff(1:3072,xind_R); 


    Xbar_R = [];
    Xbar_Q = [];
    Xbar_hat = [];

    location_to_R_indices = struct;
    location_to_Q_indices = struct ;
    location_to_Xhat_indices = struct;

   

    % For all the directories in Locations/
    N = length(directories);
    
    valid_location_indices = [];
    
    for location_index = 1 : N
      
         if(strcmp(directories(location_index).name,'.') == 0 && ...
            strcmp(directories(location_index).name,'..') == 0 &&  ...
            strcmp(directories(location_index).name,'.DS_Store') == 0)

               valid_location_indices = [valid_location_indices,location_index];

               location_to_R_indices.(directories(location_index).name) = struct;
               location_to_Q_indices.(directories(location_index).name) = struct;
               location_to_Xhat_indices.(directories(location_index).name) = struct;
               


         end

    end
    
    
    for location_index = valid_location_indices
        
            Xall = load(char(strcat(directories(location_index).folder,"/",directories(location_index).name,"/Xall.mat")));
            Xall = Xall.Xall;

            [dimX,numPatches] = size(Xall.data{1});
            numImages = length(Xall.data);
            used_images = floor(numImages/3);
            
          
            for ii=1:used_images %%only use first 4 images to extract training patches from
                                                   
                [impath,imname,ext] = fileparts(Xall.imfiles{ii});
                
               
                new_indices_R = [length(Xbar_R) : length(xind_R) + length(Xbar_R) - 1];
                new_indices_Q = [length(Xbar_Q) : length(xind_Q) + length(Xbar_Q) - 1];
                
                Xbar_R = [Xbar_R, Xall.data{ii}(:,xind_R)];
                Xbar_Q = [Xbar_Q, Xall.data{ii}(:,xind_Q)];
                   
                location_to_R_indices.(directories(location_index).name).(imname) = new_indices_R ;            
                location_to_Q_indices.(directories(location_index).name).(imname) = new_indices_Q ;            

                              
            end
            
            for ii = used_images + 1 : numImages
                               

                [impath,imname,ext] = fileparts(Xall.imfiles{ii});

                new_indices_Xhat = [length(Xbar_hat) : 3072 + length(Xbar_hat) - 1];

                location_to_Xhat_indices.(directories(location_index).name).(imname) = new_indices_Xhat ;            

                
                
                Xbar_hat = [Xbar_hat,Xall.data{ii}];
                
            end
            
        
                

    end

    Xbar_R = Xbar_R'; %%flip for Isomap: row i is now 1x175 dim vector
    Xbar_Q = Xbar_Q';
    Xbar_hat = Xbar_hat';
    
    
    disp(size(Xbar_R));
    
    if ( max(size(Xbar_R)) < 150)
        error('Error: Xbar too small. Add more images or increase Nsamps');
    end

    %% 1. Run Isomap
    %Compute the Euclidean distance matrix for all data in Xbar:
    %the ith row and jth column of Dbar is the Euclidean distance between
    %Xbar(i,:) and Xbar(j,:). 
    Dbar = squareform(pdist(Xbar_R));
    
    options.dims = 8; %%Ramos paper suggests 8 is sufficient
    knearest = 10; %%8-10 seems good
    Ybar_R=Isomap(Dbar,'k',knearest,options);
   
    %%%Ybar.coords is cell array with coords for d-dim embeddings in Ybar.coords{d}.  
    %%%Ybar.index has the indices of points embedded (corresponding row of Xbar).
    
    save ../../data/training/Ybar_R.mat -mat Ybar_R
    save ../../data/training/Xbar_R.mat -mat Xbar_R
    save ../../data/training/Xbar_Q.mat -mat Xbar_Q
    save ../../data/training/Xbar_hat.mat -mat Xbar_hat

    
    save ../../data/training/location_to_R_indices.mat -mat location_to_R_indices
    save ../../data/training/location_to_Q_indices.mat -mat location_to_Q_indices
    save ../../data/training/location_to_Xhat_indices.mat -mat location_to_Xhat_indices
   

    

