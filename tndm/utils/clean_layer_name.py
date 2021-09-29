def clean_layer_name(name: str) -> str:
    tks = name.split('/')[-3:-1]
    if len(tks) < 2:
        return tks[0].replace('_', '')
    else:
        return tks[0].split('_')[-1] + '_' + tks[1].replace('_', '')