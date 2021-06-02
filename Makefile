conda-create:
	--conda env create -f ./environment.yml

conda-update:
	--conda env update --name latentneural -f ./environment.yml

jupyter-add-kernel:
	--python -m ipykernel install --user --name=latentneural