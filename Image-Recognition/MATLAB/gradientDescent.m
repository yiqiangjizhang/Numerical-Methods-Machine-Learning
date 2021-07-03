function [] = gradientDescent(Xtrain, Ytrain, alpha, iterations, nodes, nr_correct, m_original)
    %{
    Gradient Descent
    %}
    [W1, b1, W2, b2] = initParameters(Xtrain, nodes);

    for i = 1:1:iterations

        for j = 1:size(Xtrain, 1)
            %[Z1, A1, ~, A2] = forwardPropagation(W1, b1, W2, b2, Xtrain(i,:)');
            % [dW1, db1, dW2, db2] = backwardPropagation(Z1, A1, A2, W2, Xtrain, Ytrain, j, W1, b1, b2, alpha);
            %[W1, b1, W2, b2] = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);

            %{
            Forward propagation
            %}
            Z1 = W1 * Xtrain(j, :)' + b1;
            % A1 = ReLU(Z1);
            A1 = Sigmoid(Z1);
            Z2 = W2 * A1 + b2;
            % A2 = softmax(Z2);
            A2 = Sigmoid(Z2);

            % Cost / Error
            m = length(Ytrain);
            one_hot_Y = onehot(Ytrain(j));
            error_MSE = 1 / m * sum((A2 - one_hot_Y).^2); % MSE
            
            [~, A2_index] = max(A2);
            [~, onehot_index] = max(one_hot_Y);

            nr_correct = nr_correct + (A2_index == onehot_index);
            
            %{
            Backward propagation
            %}

            delta_o = A2 - one_hot_Y;

            W2 = W2 - alpha * delta_o * A1';
            b2 = b2 - alpha * delta_o;

            delta_h = W2' * (delta_o .* Sigmoid_deriv(A1));

            % W1 = W2 - alpha * delta_h * Xtrain(j, :)';
            W1 = W1 - alpha * delta_h * Xtrain(j, :);
            b1 = b1 - alpha * delta_h;

            %{
            Accuracy
            %}

        end

        if mod(i, 1) == 0
            accuracy = (nr_correct / m_original) * 100;
            fprintf('Accuracy: %f \n', accuracy);
        end

        nr_correct = 0;
    end

    
    
    plot(mod(i, 1),accuracy);
    
    
end
