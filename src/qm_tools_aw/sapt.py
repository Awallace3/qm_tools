import pandas

def compute_sapt_s_inf(df,sapt_alpha=1.0, df_type="SAPT(DFT)"):
    if df_type == "SAPT(DFT)":
        df['tmp'] = df['SAPT EXCH-IND20,U ENERGY']
        df['SAPT EXCH-IND20,U ENERGY'] = df['SAPT EXCH-IND20,U (S^INF) ENERGY']
        df['SAPT HF(2) ENERGY'] = 0
        df = compute_sapt_terms(df, sapt_alpha=sapt_alpha)
        df['SAPT EXCH-IND20,U ENERGY'] = df['tmp']
    return df


def compute_sapt_terms(df, sapt_alpha=None):
    """
    Computes the SAPT variables from fundamental components.
    """
    if sapt_alpha is not None:
        df['SAPT ALPHA'] = [sapt_alpha for i in range(len(df))]

    x = ['SAPT EXCH10 ENERGY', 'SAPT EXCH10(S^2) ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT EXCHSCAL1'] = df.apply( lambda r: 1.0 if r[x[0]] < 1.0e-5 else r[x[0]] / r[x[1]], axis=1)

    x = ['SAPT EXCHSCAL1']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT EXCHSCAL3'] = df.apply(lambda r: r[x[0]] ** 3, axis=1)

    x = ['SAPT EXCHSCAL1', "SAPT ALPHA"]
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT EXCHSCAL'] = df.apply(lambda r: r[x[0]] ** r[x[1]], axis=1)

    x = ['SAPT HF TOTAL ENERGY', 'SAPT ELST10,R ENERGY', 'SAPT EXCH10 ENERGY', 'SAPT IND20,R ENERGY', 'SAPT EXCH-IND20,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT HF(2) ALPHA=0.0 ENERGY'] = df.apply(lambda r: r[x[0]] - (r[x[1]] + r[x[2]] + r[x[3]] + r[x[4]]), axis=1)

    x = ['SAPT HF TOTAL ENERGY', 'SAPT ELST10,R ENERGY', 'SAPT EXCH10 ENERGY', 'SAPT IND20,U ENERGY', 'SAPT EXCH-IND20,U ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT HF(2),U ALPHA=0.0 ENERGY'] = df.apply(lambda r: r[x[0]] - (r[x[1]] + r[x[2]] + r[x[3]] + r[x[4]]), axis=1)

    x = ['SAPT EXCHSCAL', 'SAPT HF(2),U ALPHA=0.0 ENERGY', 'SAPT EXCH-IND20,U ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT HF(2),U ENERGY'] = df.apply(lambda r: r[x[1]] + (1.0 - r[x[0]]) * r[x[2]], axis=1)

    x = ['SAPT EXCHSCAL', 'SAPT HF(2) ALPHA=0.0 ENERGY', 'SAPT EXCH-IND20,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT HF(2) ENERGY'] = df.apply(lambda r: r[x[1]] + (1.0 - r[x[0]]) * r[x[2]], axis=1)

    x = ['SAPT EXCHSCAL', 'SAPT HF(2) ENERGY', 'SAPT IND30,R ENERGY', 'SAPT EXCH-IND30,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT HF(3) ENERGY'] = df.apply(lambda r: r[x[1]] - (r[x[2]] + r[x[0]] * r[x[3]]), axis=1)

    x = ['SAPT EXCHSCAL', 'SAPT MP2 CORRELATION ENERGY', 'SAPT ELST12,R ENERGY',  'SAPT IND22 ENERGY', 'SAPT DISP20 ENERGY', 'SAPT EXCH11(S^2) ENERGY', 'SAPT EXCH12(S^2) ENERGY', 'SAPT EXCH-IND22 ENERGY', 'SAPT EXCH-DISP20 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT MP2(2) ENERGY'] = df.apply(lambda r: r[x[1]] - (r[x[2]] + r[x[3]] + r[x[4]] + r[x[0]] * (r[x[5]] + r[x[6]] + r[x[7]] + r[x[8]])), axis=1)

    x = ['SAPT EXCHSCAL', 'SAPT MP2(2) ENERGY', 'SAPT IND-DISP30 ENERGY', 'SAPT EXCH-IND-DISP30 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT MP2(3) ENERGY'] = df.apply(lambda r: r[x[1]] - (r[x[2]] + r[x[0]] * r[x[3]]), axis=1)

    x = ['SAPT EXCHSCAL', 'SAPT EXCH-DISP20 ENERGY', 'SAPT DISP20 ENERGY', 'SAPT DISP21 ENERGY', 'SAPT DISP22(SDQ) ENERGY', 'SAPT EST.DISP22(T) ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT MP4 DISP'] = df.apply(lambda r: r[x[0]] * r[x[1]] + r[x[2]] + r[x[3]] + r[x[4]] + r[x[5]], axis=1)

    x = ['SAPT EXCHSCAL', 'SAPT EXCH-DISP20 ENERGY', 'SAPT DISP2(CCD) ENERGY', 'SAPT DISP22(S)(CCD) ENERGY', 'SAPT EST.DISP22(T)(CCD) ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT CCD DISP'] = df.apply(lambda r:r[ x[0]] *r[ x[1]] +r[ x[2]] +r[ x[3]] +r[ x[4]], axis=1)

    x = ['SAPT ELST10,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT0 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis=1)

    x = ['SAPT EXCH10 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT0 EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis=1 )

    x = ['SAPT EXCHSCAL', 'SAPT HF(2) ENERGY', 'SAPT IND20,R ENERGY', 'SAPT EXCH-IND20,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT0 IND ENERGY'] = df.apply(lambda r:r[ x[1]] +r[ x[2]] +r[ x[0]] *r[ x[3]], axis=1)

    x = ['SAPT EXCHSCAL', 'SAPT HF(2),U ENERGY', 'SAPT IND20,U ENERGY', 'SAPT EXCH-IND20,U ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT0 IND,U ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[2]] + r[x[0]] * r[x[3]], axis=1)

    x = ['SAPT EXCHSCAL', 'SAPT EXCH-DISP20 ENERGY', 'SAPT DISP20 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT0 DISP ENERGY'] = df.apply(lambda r: r[x[0]] * r[x[1]] + r[x[2]], axis = 1)

    x = ['SAPT0 ELST ENERGY', 'SAPT0 EXCH ENERGY', 'SAPT0 IND ENERGY', 'SAPT0 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT0 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT0 ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SSAPT0 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT0 EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SSAPT0 EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT EXCHSCAL3', 'SAPT0 IND ENERGY', 'SAPT EXCH-IND20,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SSAPT0 IND ENERGY'] = df.apply(lambda r: r[x[1]] + (r[x[0]] - 1.0) * r[x[2]], axis = 1)

    x = ['SAPT EXCHSCAL3', 'SAPT0 IND,U ENERGY', 'SAPT EXCH-IND20,U ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SSAPT0 IND,U ENERGY'] = df.apply(lambda r: r[x[1]] + (r[x[0]] - 1.0) * r[x[2]], axis = 1)

    x = ['SAPT EXCHSCAL3', 'SAPT EXCH-DISP20 ENERGY', 'SAPT DISP20 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SSAPT0 DISP ENERGY'] = df.apply(lambda r: r[x[0]] * r[x[1]] + r[x[2]], axis = 1)

    x = ['SSAPT0 ELST ENERGY', 'SSAPT0 EXCH ENERGY', 'SSAPT0 IND ENERGY', 'SSAPT0 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SSAPT0 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT0 ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SCS-SAPT0 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT0 EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SCS-SAPT0 EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT0 IND ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SCS-SAPT0 IND ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT0 IND,U ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SCS-SAPT0 IND,U ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = [0.66, 'SAPT SAME-SPIN EXCH-DISP20 ENERGY', 'SAPT SAME-SPIN DISP20 ENERGY', 1.2, 'SAPT EXCH-DISP20 ENERGY', 'SAPT DISP20 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SCS-SAPT0 DISP ENERGY'] = df.apply(lambda r: (x[0] - x[3]) * (r[x[1]] + r[x[2]]) + x[3] * (r[x[4]] + r[x[5]]), axis=1)

    x = ['SCS-SAPT0 ELST ENERGY', 'SCS-SAPT0 EXCH ENERGY', 'SCS-SAPT0 IND ENERGY', 'SCS-SAPT0 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SCS-SAPT0 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT ELST10,R ENERGY', 'SAPT ELST12,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT EXCH10 ENERGY', 'SAPT EXCH11(S^2) ENERGY', 'SAPT EXCH12(S^2) ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2 EXCH ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[0]] * (r[x[2]] + r[x[3]]), axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT HF(2) ENERGY', 'SAPT IND20,R ENERGY', 'SAPT EXCH-IND20,R ENERGY', 'SAPT IND22 ENERGY', 'SAPT EXCH-IND22 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2 IND ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[2]] + r[x[0]] * r[x[3]] + r[x[4]] + r[x[0]] * r[x[5]], axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT EXCH-DISP20 ENERGY', 'SAPT DISP20 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2 DISP ENERGY'] = df.apply(lambda r: r[x[0]] * r[x[1]] + r[x[2]], axis = 1)

    x = ['SAPT2 ELST ENERGY', 'SAPT2 EXCH ENERGY', 'SAPT2 IND ENERGY', 'SAPT2 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT ELST10,R ENERGY', 'SAPT ELST12,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+ ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT EXCH10 ENERGY', 'SAPT EXCH11(S^2) ENERGY', 'SAPT EXCH12(S^2) ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+ EXCH ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[0]] * (r[x[2]] + r[x[3]]), axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT HF(2) ENERGY', 'SAPT IND20,R ENERGY', 'SAPT EXCH-IND20,R ENERGY', 'SAPT IND22 ENERGY', 'SAPT EXCH-IND22 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+ IND ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[2]] + r[x[0]] * r[x[3]] + r[x[4]] + r[x[0]] * r[x[5]], axis = 1)

    x = ['SAPT MP4 DISP']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+ DISP ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+ ELST ENERGY', 'SAPT2+ EXCH ENERGY', 'SAPT2+ IND ENERGY', 'SAPT2+ DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+ TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+ ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(CCD) ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+ EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(CCD) EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+ IND ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(CCD) IND ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT CCD DISP']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(CCD) DISP ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(CCD) ELST ENERGY', 'SAPT2+(CCD) EXCH ENERGY', 'SAPT2+(CCD) IND ENERGY', 'SAPT2+(CCD) DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(CCD) TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+ ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+DMP2 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+ EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+DMP2 EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+ IND ENERGY', 'SAPT MP2(2) ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+DMP2 IND ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+ DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+DMP2 DISP ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+DMP2 ELST ENERGY', 'SAPT2+DMP2 EXCH ENERGY', 'SAPT2+DMP2 IND ENERGY', 'SAPT2+DMP2 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+DMP2 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+ ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(CCD)DMP2 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+ EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(CCD)DMP2 EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+DMP2 IND ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(CCD)DMP2 IND ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(CCD) DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(CCD)DMP2 DISP ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(CCD)DMP2 ELST ENERGY', 'SAPT2+(CCD)DMP2 EXCH ENERGY', 'SAPT2+(CCD)DMP2 IND ENERGY', 'SAPT2+(CCD)DMP2 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(CCD)DMP2 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT ELST10,R ENERGY', 'SAPT ELST12,R ENERGY', 'SAPT ELST13,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3) ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT EXCH10 ENERGY', 'SAPT EXCH11(S^2) ENERGY', 'SAPT EXCH12(S^2) ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3) EXCH ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[0]] * (r[x[2]] + r[x[3]]), axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT HF(2) ENERGY', 'SAPT IND20,R ENERGY', 'SAPT EXCH-IND20,R ENERGY', 'SAPT IND22 ENERGY', 'SAPT EXCH-IND22 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3) IND ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[2]] + r[x[0]] * r[x[3]] + r[x[4]] + r[x[0]] * r[x[5]], axis = 1)

    x = ['SAPT MP4 DISP', 'SAPT DISP30 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3) DISP ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3) ELST ENERGY', 'SAPT2+(3) EXCH ENERGY', 'SAPT2+(3) IND ENERGY', 'SAPT2+(3) DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3) TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3) ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)(CCD) ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3) EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)(CCD) EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3) IND ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)(CCD) IND ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT CCD DISP', 'SAPT DISP30 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)(CCD) DISP ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3)(CCD) ELST ENERGY', 'SAPT2+(3)(CCD) EXCH ENERGY', 'SAPT2+(3)(CCD) IND ENERGY', 'SAPT2+(3)(CCD) DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)(CCD) TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3) ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)DMP2 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3) EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)DMP2 EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3) IND ENERGY', 'SAPT MP2(2) ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)DMP2 IND ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3) DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)DMP2 DISP ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3)DMP2 ELST ENERGY', 'SAPT2+(3)DMP2 EXCH ENERGY', 'SAPT2+(3)DMP2 IND ENERGY', 'SAPT2+(3)DMP2 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)DMP2 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3) ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)(CCD)DMP2 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3) EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)(CCD)DMP2 EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3)DMP2 IND ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)(CCD)DMP2 IND ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3)(CCD) DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)(CCD)DMP2 DISP ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+(3)(CCD)DMP2 ELST ENERGY', 'SAPT2+(3)(CCD)DMP2 EXCH ENERGY', 'SAPT2+(3)(CCD)DMP2 IND ENERGY', 'SAPT2+(3)(CCD)DMP2 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+(3)(CCD)DMP2 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT ELST10,R ENERGY', 'SAPT ELST12,R ENERGY', 'SAPT ELST13,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT EXCH10 ENERGY', 'SAPT EXCH11(S^2) ENERGY', 'SAPT EXCH12(S^2) ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3 EXCH ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[0]] * (r[x[2]] + r[x[3]]), axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT HF(3) ENERGY', 'SAPT IND20,R ENERGY', 'SAPT EXCH-IND20,R ENERGY', 'SAPT IND22 ENERGY', 'SAPT EXCH-IND22 ENERGY', 'SAPT IND30,R ENERGY', 'SAPT EXCH-IND30,R ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3 IND ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[2]] + r[x[0]] * r[x[3]] + r[x[4]] + r[x[0]] * r[x[5]] + r[x[6]] + r[x[0]] * r[x[7]], axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT MP4 DISP', 'SAPT DISP30 ENERGY', 'SAPT EXCH-DISP30 ENERGY', 'SAPT IND-DISP30 ENERGY', 'SAPT EXCH-IND-DISP30 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3 DISP ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[2]] + r[x[0]] * r[x[3]] + r[x[4]] + r[x[0]] * r[x[5]], axis = 1)

    x = ['SAPT2+3 ELST ENERGY', 'SAPT2+3 EXCH ENERGY', 'SAPT2+3 IND ENERGY', 'SAPT2+3 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3 ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3(CCD) ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3 EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3(CCD) EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3 IND ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3(CCD) IND ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT EXCHSCAL', 'SAPT CCD DISP', 'SAPT DISP30 ENERGY', 'SAPT EXCH-DISP30 ENERGY', 'SAPT IND-DISP30 ENERGY', 'SAPT EXCH-IND-DISP30 ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3(CCD) DISP ENERGY'] = df.apply(lambda r: r[x[1]] + r[x[2]] + r[x[0]] * r[x[3]] + r[x[4]] + r[x[0]] * r[x[5]], axis = 1)

    x = ['SAPT2+3(CCD) ELST ENERGY', 'SAPT2+3(CCD) EXCH ENERGY', 'SAPT2+3(CCD) IND ENERGY', 'SAPT2+3(CCD) DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3(CCD) TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3 ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3DMP2 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3 EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3DMP2 EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3 IND ENERGY', 'SAPT MP2(3) ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3DMP2 IND ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3DMP2 DISP ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3DMP2 ELST ENERGY', 'SAPT2+3DMP2 EXCH ENERGY', 'SAPT2+3DMP2 IND ENERGY', 'SAPT2+3DMP2 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3DMP2 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3 ELST ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3(CCD)DMP2 ELST ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3 EXCH ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3(CCD)DMP2 EXCH ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3DMP2 IND ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3(CCD)DMP2 IND ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3(CCD) DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3(CCD)DMP2 DISP ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    x = ['SAPT2+3(CCD)DMP2 ELST ENERGY', 'SAPT2+3(CCD)DMP2 EXCH ENERGY', 'SAPT2+3(CCD)DMP2 IND ENERGY', 'SAPT2+3(CCD)DMP2 DISP ENERGY']
    if set([i for i in x if type(i) == str]).issubset(df.columns):
        df['SAPT2+3(CCD)DMP2 TOTAL ENERGY'] = df.apply(lambda r: sum([r[i] for i in x]), axis = 1)

    return df
