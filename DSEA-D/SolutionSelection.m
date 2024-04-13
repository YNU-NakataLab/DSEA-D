function Offspring = SolutionSelection(Problem, Candidates, model, R_max, i, W, Z, mode)
% Solution Selection in DSEA/D

%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yuma Horaguchi

    switch mode
        case 'approx'
            fpred       = transpose(sim(model, transpose(Candidates)));
            gpred       = max(abs(fpred - repmat(Z, length(R_max), 1)) .* W(i, :), [], 2);
            [~, sortID] = sort(gpred);
            Offspring   = Candidates(sortID(1), :);
        case 'class'
            UniformedCandidates = (Candidates - Problem.lower) ./ (Problem.upper - Problem.lower);
            [class, score]      = model.predict(UniformedCandidates);
            score               = score(:, 2);
            if nnz(class == +1)
                goodID          = find(class == +1);
                Offspring       = Candidates(goodID(1), :);
                class           = +1;
            else
                [~, goodID]     = sort(score, 'descend');
                Offspring       = Candidates(goodID(1), :);
                class           = -1;
            end
    end
end