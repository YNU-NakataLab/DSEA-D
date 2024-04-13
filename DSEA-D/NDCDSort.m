function [PopSrt, SrtID, FrontNo, CrowdDis] = NDCDSort(varargin)
% Sort the population based on non-dominated sorting and crowding distance

%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if nargin == 1
        Population = varargin{1};
        if isa(Population(1),'SOLUTION')
            FrontNo    = NDSort(Population.objs, Population.cons, inf);
            CrowdDis   = CrowdingDistance(Population.objs, FrontNo);
            [~, SrtID] = sortrows([FrontNo; -CrowdDis]');
            PopSrt     = Population(SrtID);
        else
            PopObj     = varargin{1};
            FrontNo    = NDSort(PopObj, inf);
            CrowdDis   = CrowdingDistance(PopObj, FrontNo);
            [~, SrtID] = sortrows([FrontNo; -CrowdDis]');
            PopSrt     = PopObj(SrtID, :);
        end
    else
        PopObj     = varargin{1};
        PopCon     = varargin{2};
        FrontNo    = NDSort(PopObj, PopCon, inf);
        CrowdDis   = CrowdingDistance(PopObj, FrontNo);
        [~, SrtID] = sortrows([FrontNo; -CrowdDis]');
        PopSrt     = PopObj(SrtID, :);
    end
end
