classdef fxp_svm
    properties
        C;
        W;
        lr;
    end
    
    methods
        function obj = fxp_svm(CParam, LearningRate)
           obj.lr = LearningRate;
           obj.C = CParam;
        end
        
        function obj = fit(obj, X, y, epochs)
            % Number of features
            m = size(X, 2);
            
            % Append column of ones at the end of X
            ones_vec = ones(size(X, 1), 1);
            data = [X, ones_vec];
            y_data = y(:, :);
            
            obj.W = rand(m + 1, 1);
            
            for epoch = 1:epochs
                ix_shuffle = randperm(size(X, 1));
                data = data(ix_shuffle, :);
                y_data = y_data(ix_shuffle, :);
                
                grad = obj.compute_grads(data, y_data);
                obj.W = obj.W - obj.lr .* grad;
                disp(["cost", obj.compute_loss(data, y_data)])
            end                     
        end
        
        function prediction = predict(obj, X)
           ones_vec = ones(size(X, 1), 1);
           data = [X, ones_vec];
           prediction = sign(data * obj.W);
        end
        
        function loss = compute_loss(obj, X, y)
            N = size(X, 1);
            distances = 1 - y .* (X * obj.W);
            distances(distances < 0) = 0;
            
            % Compute Hinge loss
            hinge_loss = obj.C .* sum(distances) ./ N;
            
            % Calculate cost
            loss = (1 / 2) .* (obj.W' * obj.W) + hinge_loss;
        end
        
        function grads = compute_grads(obj, X, y)
           distance = ones(length(y), 1) - y .* (X * obj.W);
           grads = zeros(size(obj.W, 1, 2));
           
           for index = 1:length(distance)
               grads = zeros(size(obj.W, 1, 2));
               if max(0, distance(index)) == 0
                   dist_i = obj.W;
               else
                   dist_i = obj.W - obj.C .* y(index) .* X(index,:)';
               end
               grads = grads + dist_i;
           end
           
           grads = grads ./ size(X, 1);
        end
        
        function sc = score(obj, X, y_true)
           prediction = obj.predict(X);
           sc = sum(y_true == prediction) / size(X, 1);
        end
    end
end
