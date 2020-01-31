function w = LearnToCount(nFeatures,features,weights,weightMask,densities,...
                          C,maxIter,verbose)
%LearnToCount - implements the "Learning To Count Objects in Images"//NIPS 2010 framework
%INPUTS
%nFeatures = dimensionality of the feature vector
%features = cell array of [M x N x L] arrays corresponding to non-zero
%      feature coordinate in (M x N) image grids with L non-zero features
%      per pixel;      each cell entry corresponds to the training instance
%weights = same structureas above, but indicates the values of these non-zero
%      features
%weightMask = an [M x N] array that modulates feature weights, was set to
%      all ones for cell counting and to inverse squared depth for crowd counting 
%densities = ground truth densities to match. A cell array of [M x N] arrays
%C =   weight for the data component in the learning objective. Positive
%      values would lead to Tikhonov regularization (as in the paper).
%      Negative values leads to L1 regularization used (the absolute value
%      of C is used)
%maxIter = number of iterations after which to break. The process would
%      stop earlier if the constraints are satisfied upto eps = 0.01
%verbose = displays extra information and pictures
%
%OUTPUT: w = the learned weight vector for counting.
%NB: if weights are non-negative, the code would also impose
%           non-negativity constraint on w.
%
% (C) Victor Lempitsky, 2011

imSize = [size(features{1},1) size(features{1},2)];
nImages = numel(features);
initial = 20;
preAlloc = 1+initial+maxIter*nImages*2;
rects = zeros(preAlloc,5);
nRects = 0;

A = sparse(2*preAlloc,nFeatures+nImages);
b = zeros(2*preAlloc,1);

%draw some rectangles randomly
for r = 1:initial
   x = zeros(2,1);
   y = zeros(2,1);
   while abs(x(2) - x(1))<1 || abs(y(2) - y(1))<1
        x = ceil(rand(2,1)*imSize(2)); 
        y = ceil(rand(2,1)*imSize(1)); 
        i = ceil(rand*nImages);
   end
   AddRect([min(x) max(x)],[min(y) max(y)],i);
end

if C >= 0 
    disp('Tikhonov regularization is used.');
    H = speye(nFeatures+nImages);
    H(nFeatures+1:end,nFeatures+1:end) = 0;
    f = [zeros(nFeatures,1); C*ones(nImages,1)];
else
    disp('L1 regularization is used.');
    f = [ones(nFeatures,1); -C*ones(nImages,1)];
end


if all(all(all(cat(1,weights{:})>=0))) % replace with 'if 0' to avoid non-negativity constraint on w
    lb = [zeros(nFeatures,1); zeros(nImages,1)]; 
    disp('Non-negative feature encoding assumed. w is constrained to be non-negative.');
else
    disp('Negative feature weights detected. No constraints on the sign of w imposed.');
    lb = [-inf(nFeatures,1); zeros(nImages,1)];
    if C < 0
        error('Negative weights and L1 regularization not supported at the moment.');
    end
end


x = [];

for iter = 1:maxIter 
   disp(['Starting iteration ' num2str(iter)]);
   if C >= 0
       x = quadprog(H,f,A(1:nRects*2,:),b(1:nRects*2),[],[],lb,[],x);
   else
       x = linprog(f,A(1:nRects*2,:),b(1:nRects*2),[],[],lb,[]);
   end
   if isempty(x)
       error('Error in convex solver. Most likely - out of memory.');
   end   
   
   w = x(1:nFeatures);
   slacks = x(nFeatures+1:end);
   changed = 0;
   
   disp('Convex program (re)solved. Generating constraints...');
   for im = 1:nImages
       density = w(features{im}).*double(weights{im});
       density = sum(density,3).*weightMask;
       diff = density-densities{im};
   
       [ymin1 ymax1 xmin1 xmax1 val1] = maxsubarray2D(diff);
       [ymin2 ymax2 xmin2 xmax2 val2] = maxsubarray2D(-diff);
   
       slack = slacks(im);
       
       if max(val1,val2) < slack*1.01 %the epsilon = 0.01 can be tweaked
           continue;
       end

       changed = 1;
       if val1 > val2
           AddRect([xmin1 xmax1], [ymin1 ymax1], im);
           x(nFeatures+im) = val1+1e-6;
       else
           AddRect([xmin2 xmax2], [ymin2 ymax2], im);
           x(nFeatures+im) = val2+1e-6;
       end
       
      if verbose && im == 1
           disp(['Statistics for image 1: True count: ' num2str(sum(densities{im}(:)),'%.1f') '; Estimated count: ' num2str(sum(density(:)), '%.1f')...
               '; MESA-distance: ' num2str(max(val1,val2),'%.1f') '; Previous slack: ' num2str(slack,'%.1f') ]);
           disp('Showing the GT density, the estimated density, the diff and the most violated constraint');
           set(gcf, 'Color', 'w');
           subplot(1,3,1); 
           imagesc(densities{im}); daspect([1 1 1]); axis off;
           subplot(1,3,2); 
           imagesc(density); daspect([1 1 1]); axis off;
           subplot(1,3,3); 
           imagesc(diff); daspect([1 1 1]); hold on; axis off;
           rectangle('Position',[rects(nRects,1) rects(nRects,3)...
               rects(nRects,2)-rects(nRects,1) rects(nRects,4)-rects(nRects,3)],...
               'LineWidth',2, 'EdgeColor', 'm');
           hold off;
           drawnow;
       end             
       
       if nRects == preAlloc
           warning('Number of pre-allocated rectangles reached! Terminating the constraint generation');
           break;
       end
       
    end
   if changed == 0
       disp('Constraint generation successfully converged');
       break;
   end
end

    function AddRect(x,y,i)
       nRects = nRects+1;
       rects(nRects,:) = [x(:)' y(:)' i];
       value = sum(sum(densities{i}(min(y):max(y),min(x):max(x))));
       roi = double(features{i}(min(y):max(y),min(x):max(x),:));
       weight = double(repmat(weightMask(min(y):max(y),min(x):max(x)), [1 1 size(roi,3)])).*...
           double(weights{i}(min(y):max(y),min(x):max(x),:));
       feature = accumarray(roi(:), double(weight(:)), [nFeatures, 1]);
       A(2*nRects-1,:) = [feature' zeros(1,nImages)];
       A(2*nRects-1,nFeatures+i) = -1;
       b(2*nRects-1) = value;
       A(2*nRects,:) = [-feature' zeros(1,nImages)];
       A(2*nRects,nFeatures+i) = -1;
       b(2*nRects) = -value;       
    end
end 

   

