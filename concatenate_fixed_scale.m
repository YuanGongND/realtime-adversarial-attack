function [] = concatenate_fixed_scale(base_original_path, base_perturbed_path, target_path)

% initialize the files
original_data = zeros([25000, 16000]);
perturbation_data = zeros([25000, 500]);
label_info = zeros([25000, 1]);
entry_index = 1;

% the ten word list
word_list = {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'};

for word_index = 1: 10
    current_label = word_index + 1; % consistant with other
    
    original_path = [base_original_path, '/', word_list{word_index}];
    perturbed_path = [base_perturbed_path, '/', word_list{word_index}];
    perturbed_file_list = dir(perturbed_path);

    for file_id = 3: length(perturbed_file_list)
       filename = perturbed_file_list(file_id).name; 
        
       % if is wav file
       if filename(end-2: end) == 'wav'
           original_filename = [filename(1:end-14), '.wav'];
           perturbed_info_filename = [perturbed_path, '/', original_filename(1: end-4), '.csv'];

           if isfile([original_path, '/', original_filename])
              perturbed_info = csvread(perturbed_info_filename);
              perturbed_info_onehot = convert_to_onehot_5(perturbed_info);
              original_wav = audioread([original_path, '/', original_filename]);

              original_data(entry_index, 1: length(original_wav')) = original_wav';
              perturbation_data(entry_index, :) = perturbed_info_onehot;
              label_info(entry_index, :) = current_label;
              entry_index = entry_index + 1
           else
               disp('error');
           end
       end
    end
end

csvwrite([target_path, '/original.csv'], original_data(1:entry_index - 1, :));
csvwrite([target_path, '/perturbation.csv'], perturbation_data(1:entry_index - 1, :));
csvwrite([target_path, '/label_info.csv'], label_info(1:entry_index - 1, :));

end
   
function [perturb_info_onehot] = convert_to_onehot_5(perturb_info)
    perturb_info_onehot = zeros([1, 500]);
    [~, sort_position] = sort(perturb_info([1,2,3,4,5]));
    for i = 1: 5
        perturb_info_onehot(1, (i - 1) * 100 + 1: (i - 1) * 100 + 100) = perturb_info(sort_position(i)) / 100;
    end
end