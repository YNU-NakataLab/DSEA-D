function [Dvalid, Dtrain] = DatasetBuilder(Problem, Candidates, Arc)
% Dataset builder in DSEA/D

%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yuma Horaguchi

    distance = pdist2(Candidates, Arc.decs);
    N_i      = [];
    for k = 1 : size(Candidates, 1)
        [~, NearID] = min(distance(k, :));
        N_i         = [N_i, NearID];
        distance(:, NearID) = Inf;
    end
    Dvalid        = Arc(N_i);
    [~, RemainID] = setdiff(Arc.decs, Dvalid.decs, 'stable', 'rows');
    ArcRem        = Arc(RemainID);
    AremainNDS    = NDCDSort(ArcRem);
    Dtrain        = AremainNDS(1 : min(Problem.N, length(ArcRem)));
end