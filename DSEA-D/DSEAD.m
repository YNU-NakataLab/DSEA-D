classdef DSEAD < ALGORITHM
% <multi/many> <real/integer> <expensive>
% Dual surrogate-based evolutionary algorithm based on decomposition
% delta  --- 0.9 --- The probability of choosing parents locally
% nr     ---   2 --- Maximum number of solutions replaced by each offspring
% Rmax   ---  10 --- Maximum repeat time of offspring generation
% C      --- 1.0 --- SVM parameter C
% gamma  --- 1.0 --- SVM parameter gamma
% theta  --- 0.5 --- The threshold for Kendall tau

%------------------------------- Reference --------------------------------
% Y. Horaguchi and M. Nakata, A Dual Surrogate-based Evolutionary Algorithm
% for High-Dimensional Expensive Multiobjective Optimization Problems, 
% Proceedings of the IEEE Congress on Evolutionary Computation, 2024.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Yuma Horaguchi

    methods
        function main(Algorithm, Problem)
            warning('off');

            %% Parameter setting
            [delta, nr, R_max, C, gamma, theta] = Algorithm.ParameterSet(0.9, 2, 10, 1.0, 1.0, 0.5);

            %% Generate the weight vectors
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);

            %% Detect the neighbours of each solution
            T      = ceil(Problem.N / 10);
            B      = pdist2(W, W);
            [~, B] = sort(B, 2);
            B      = B(:, 1 : T);

            %% Initialize population
            PopDec      = UniformPoint(Problem.N, Problem.D, 'Latin');
            Population  = Problem.Evaluation(repmat(Problem.upper - Problem.lower, Problem.N, 1) .* PopDec + repmat(Problem.lower, Problem.N, 1));
            Arc         = Population;
            Z           = min(Population.objs, [], 1);
            sigma       = sqrt(1 / (2 * gamma));

            %% Optimization
            while Algorithm.NotTerminated(Arc)
                % For each sub-problem
                for i = 1 : Problem.N
                    %% Remove duplicated solution from archive
                    [~, UniqueID] = unique(Arc.decs, 'stable', 'rows');
                    ArcUniqe      = Arc(UniqueID);

                    %% Choose the parents
                    if rand < delta
                        P = B(i, randperm(end));
                    else
                        P = randperm(Problem.N);
                    end

                    %% Generate candidate set
                    Candidates = [];
                    for r = 1 : R_max
                        Candidate  = OperatorDE(Problem, Population(i).dec, Population(P(1)).dec, Population(P(2)).dec);
                        rnd        = randperm(length(P));
                        P          = P(rnd);
                        Candidates = [Candidates; Candidate];
                    end

                    %% Construct and validation of approximation models
                    [Dvalid, Dtrain] = DatasetBuilder(Problem, Candidates, ArcUniqe);

                    %% Construct RBFN models
                    D_max      = max(max(pdist2(Dtrain.decs, Dtrain.decs), [], 2));
                    spread     = D_max * (Problem.D * Problem.N) ^ (-1 / Problem.D);
                    rbf_mdl    = newrbe(transpose(Dtrain.decs), transpose(Dtrain.objs), spread);

                    %% Evaluate the model accuracy of RBFN models
                    gtrue      = max(abs(Dvalid.objs - repmat(Z, length(Dvalid), 1)) .* W(i, :), [], 2);
                    fpred      = transpose(sim(rbf_mdl, transpose(Dvalid.decs)));
                    gpred      = max(abs(fpred - repmat(Z, length(Dvalid), 1)) .* W(i, :), [], 2);
                    tau        = corr(gtrue, gpred, 'type', 'Kendall');

                    %% Adaptive selection of approximation and classification models
                    if tau >= theta
                        % Approximation model based solution selection
                        OffDec = SolutionSelection(Problem, Candidates, rbf_mdl, R_max, i, W, Z, 'approx');
                    else
                        % Construct an SVM classifier
                        C_i   = [];
                        label = -1 * ones(length(Arc), 1);
                        for k = 1 : length(B(i, :))
                            g_A        = max(abs(Arc.objs - repmat(Z, length(Arc), 1)) .* W(B(i, k), :), [], 2);
                            [~, rankA] = sort(g_A);
                            for j = 1 : length(Arc)
                                if ~ismember(rankA(j), C_i)
                                    C_i = [C_i, rankA(j)];
                                    label(rankA(j)) = +1;
                                    break
                                end
                            end
                        end
                        UniformedADec = (Arc.decs - Problem.lower) ./ (Problem.upper - Problem.lower);
                        svm_mdl       = fitcsvm(UniformedADec, label, 'BoxConstraint', C, 'KernelScale', sigma, 'KernelFunction', 'rbf');

                        % Classification model based solution selection
                        OffDec = SolutionSelection(Problem, Candidates, svm_mdl, R_max, i, W, Z, 'class');
                    end

                    %% Evaluate offspring
                    Offspring = Problem.Evaluation(OffDec);

                    %% Update the reference point
                    Z = min(Z, Offspring.obj);

                    %% Update population and archive
                    g_old = max(abs(Population(P).objs - repmat(Z, length(P), 1)) .* W(P, :), [], 2);
                    g_new = max(repmat(abs(Offspring.obj - Z), length(P), 1) .* W(P, :), [], 2);
                    Population(P(find(g_old >= g_new, nr))) = Offspring;
                    Arc   = [Arc, Offspring];

                    %% Check termination criteria
                    Algorithm.NotTerminated(Arc);
                end
            end
        end
    end
end
