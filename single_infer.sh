export CUDA_VISIBLE_DEVICES=1

python inference.py \
--single_view \
--base_model_path "/data/dylu/project/see4d/dataset/checkpoint/MVD_weights/" \
--source_imgs_dir "/data/dylu/project/see4d/dataset/dataset/Kubric4D/1/reference_images/" \
--warp_root_dir "/data/dylu/project/see4d/dataset/dataset/Kubric4D/1/warp_images/" \
--output_dir "/data/dylu/project/see4d/dataset/dataset/Kubric4D/1/inpaint_images/"
done
