base_original_path = '../dataset/test2';
base_perturbed_path = '../perturbed_data/tes2_0.1_all';
target_path = '../ready_to_train_files/attack_train_pixel_5_fixed_0.1_test_all';
if ~exist(target_path,'dir')   
    mkdir(target_path);
end
concatenate_fixed_scale(base_original_path, base_perturbed_path, target_path)




