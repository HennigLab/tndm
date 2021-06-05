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