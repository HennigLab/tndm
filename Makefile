conda-create:
	--conda env create -f ./environment.yml

conda-update:
	--conda env update --name latentneural -f ./environment.yml

conda-create-cpu:
	--conda env create -f ./environment.cpu.yml

conda-update-cpu:
	--conda env update --name latentneural -f ./environment.cpu.yml

jupyter-add-kernel:
	--python -m ipykernel install --user --name=latentneural

download-kia:
	# --wget -O latentneural/data/storage/kia/metadata.json https://www.dropbox.com/s/p69wegqo4catn8f/metadata.json?dl=1
	# --wget -O latentneural/data/storage/kia/dataset.h5 https://www.dropbox.com/s/bovcnjy2f40sgpk/dataset.h5?dl=1
	--wget -O latentneural/data/storage/kia/cross-validation.zip https://www.dropbox.com/s/i48jv8qu34gbzgk/cross-validation.zip?dl=1
	--unzip -o latentneural/data/storage/kia/cross-validation.zip -d latentneural/data/storage/kia
	--rm latentneural/data/storage/kia/cross-validation.zip
	--rm -r latentneural/data/storage/kia/__MACOSX
	--wget -O latentneural/data/storage/kia/cross-validation-emg.zip https://www.dropbox.com/s/qzlp0qofms9jlwe/cross-validation-emg.zip?dl=1
	--unzip -o latentneural/data/storage/kia/cross-validation-emg.zip -d latentneural/data/storage/kia
	--rm latentneural/data/storage/kia/cross-validation-emg.zip
	--rm -r latentneural/data/storage/kia/__MACOSX
	--wget -O notebooks/results/emg_data/emg_cleansing.pdf https://www.dropbox.com/s/n1lxscl8dnb4550/emg_cleansing.pdf?dl=1

download-lorenz:
	# --mkdir latentneural/data/storage/lorenz/20210604T155502
	# --wget -O latentneural/data/storage/lorenz/20210604T155502/metadata.json https://www.dropbox.com/s/tvfl8z7k8k37vy4/metadata.json?dl=1
	# --wget -O latentneural/data/storage/lorenz/20210604T155502/dataset.h5 https://www.dropbox.com/s/6nwvbmw3r36i6wh/dataset.h5?dl=1
	--mkdir latentneural/data/storage/lorenz/20210610T215300
	--wget -O latentneural/data/storage/lorenz/20210610T215300/metadata.json https://www.dropbox.com/s/0810h1ozhiyasmm/metadata.json?dl=1
	--wget -O latentneural/data/storage/lorenz/20210610T215300/dataset.h5 https://www.dropbox.com/s/1xrfbh78de3amyd/dataset.h5?dl=1
	--wget -O latentneural/data/storage/lorenz/20210610T215300/results.zip https://www.dropbox.com/s/yk8xc4ba7mw3a6s/results.zip?dl=1
	--unzip latentneural/data/storage/lorenz/20210610T215300/results.zip -d latentneural/data/storage/lorenz/20210610T215300 -o
	--rm latentneural/data/storage/lorenz/20210610T215300/results.zip
	# --wget -O latentneural/data/storage/lorenz/grid.zip https://www.dropbox.com/s/msdf0dfi9for977/grid.zip?dl=1
	# --unzip latentneural/data/storage/lorenz/grid.zip -d latentneural/data/storage/lorenz/ -o
	# --rm latentneural/data/storage/lorenz/grid.zip
	# --wget -O latentneural/data/storage/lorenz/grid/results_old.zip https://www.dropbox.com/s/6vycfvy3ywzart1/results_old.zip?dl=1
	# --unzip -o latentneural/data/storage/lorenz/grid/results_old.zip -d latentneural/data/storage/lorenz/grid
	# --rm latentneural/data/storage/lorenz/grid/results_old.zip

tensorboard-view:
	--tensorboard dev upload --logdir latentneural/data/storage/lorenz/20210610T215300/results/lfads_log