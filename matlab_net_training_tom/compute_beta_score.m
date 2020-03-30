function [accuracy,f_measure,f_beta,g_beta] = compute_beta_score(labels, outputs,beta,num_classes)
    % Check inputs for errors.
    if length(outputs) ~= length(labels)
        error('Numbers of outputs and labels must be the same.');
    end

    [num_recordings,num_classes_from_lab] = size(labels);

        % Check inputs for errors.
    if length(num_classes) ~= length(num_classes_from_lab)
        error('Numbers of classes and labels must be the same.');
    end

    % Populate contingency table.

    fbeta_l = zeros(1,num_classes);
    gbeta_l = zeros(1,num_classes);
    fmeasure_l = zeros(1,num_classes);
    accuracy_l = zeros(1,num_classes);

    f_beta = 0;
    g_beta = 0;
    f_measure = 0;
    accuracy = 0;

    % Weigth function
    C_l = ones(1,num_classes);

    for j=1:num_classes
	tp = 0;
	fp = 0;
	fn = 0;
	tn = 0;

	for i = 1 : num_recordings

		num_labels = sum(labels(i,:));

	        if labels(i,j)==1 && outputs(i,j)==1
	            tp = tp + 1/num_labels;
	        elseif labels(i,j)~=1 && outputs(i,j)==1
	            fp = fp + 1/num_labels;
	        elseif labels(i,j)==1 && outputs(i,j)~=1
    		    fn = fn + 1/num_labels;
	        elseif labels(i,j)~=1 && outputs(i,j)~=1
	            tn = tn + 1/num_labels;
	        end
	end

	% Summarize contingency table.
        if ((1+beta^2)*tp + (beta*fn) + fp) > 0
	        fbeta_l(j) = ((1+beta^2)*tp) / ((1+beta^2)*tp + (beta^2*fn) + fp);
        else
        	fbeta_l(j) = 1;
        end

	if (tp + (beta*fn) + fp) > 0
	        gbeta_l(j) = tp / (tp + (beta*fn) + fp);
	else
	        gbeta_l(j) = 1;
	end

	if (tp + fp + fn + tn) > 0
	        accuracy_l(j) = (tp+tn) / (tp+fp+fn+tn);
	else
	        accuracy_l(j) = 1;
	end

	if (2*tp + fp + tn) >0
		fmeasure_l(j) = (2*tp)/((2*tp)+fp+fn);
	else
		fmeasure_l(j) = 1;
	end

    end

    for i = 1:num_classes
	    f_beta = f_beta + fbeta_l(i)*C_l(i);
            g_beta = g_beta + gbeta_l(i)*C_l(i);
            f_measure = f_measure + fmeasure_l(i)*C_l(i);
            accuracy = accuracy + accuracy_l(i)*C_l(i);
    end

    f_beta = f_beta/num_classes;
    g_beta = g_beta/num_classes;
    f_measure = f_measure/num_classes;
    accuracy = accuracy/num_classes;

end