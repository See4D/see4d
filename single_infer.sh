for i in {0..26}; do
python inference.py \
--single_view \
--base_model_path "./dataset/checkpoint/MVD_weights/" \
--source_imgs_dir "/dataset/htx/see4d/warps/outputs/cat_reverse_k3/frame_$i/reference_images/" \
--warp_root_dir "/dataset/htx/see4d/warps/outputs/cat_reverse_k3/frame_$i/warp_images/" \
--output_dir "/dataset/htx/see4d/outputs/cat_reverse_k3/frame_$i"
done
