
# option + (<- | ->) = move forward/backward a whole word
# command + (<- | ->) = move to beginning/end of line
# option + command + (up-arrow | down-arrow) = add cursor to line above/below

# NOTE: Ordinality of list is in reverse order: best values are first
category_ordinal_mapping = {
    'Utilities' : ['AllPub','NoSewr','NoSeWa','ELO'],
    'Exter Qual' : ['Ex','Gd','TA','Fa','Po'],
    'Exter Cond' : ['Ex','Gd','TA','Fa','Po'],
    'Bsmt Qual' : ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'Bsmt Cond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'Bsmt Exposure': ['Gd', 'Av', 'Mn', 'No', 'NA'],
    'BsmtFin Type 1': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'BsmtFinType 2': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
    'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
    'FireplaceQu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'Garage Finish': ['Fin', 'RFn', 'Unf', 'NA'],
    'Garage Qual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'Garage Cond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'Paved Drive': ['Y', 'P', 'N'],
    'Pool QC': ['Ex', 'Gd', 'TA', 'Fa', 'NA'],
    'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA'],
}