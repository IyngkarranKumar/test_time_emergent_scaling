

if 1:

    if 0: 
        from datasets import load_dataset
        dataset_name = "Maxwell-Jia/AIME_2024"

        AIME24_dataset = load_dataset(dataset_name)

        instances = [10,0,12,2]

        for instance in instances:
            print(AIME24_dataset['train'][instance])

    if 1:
        from datasets import load_dataset
        dataset_name = "Idavidrein/gpqa"
        GPQA_dataset = load_dataset(dataset_name,"gpqa_diamond")

        instances = [31,111,150,122]

        for instance in instances:
            print(GPQA_dataset['train'][instance])


if 0: 

    import numpy as np 
    import matplotlib.pyplot as plt
    from analysis.core import breakthrough_score, differences_skew_score 


    x = np.linspace(0,1,11)
    y_exp = np.exp(x) - np.exp(0)
    y_exp_halfx = np.exp(0.5 * x) - np.exp(0)
    y_exp_2x_minus_half = np.exp(2 * (x - 0.5)) - np.exp(2*(0-0.5))
    y_onestep = (x >= 0.5).astype(float)
    y_twostep = 0.5 * (x >= 0.5).astype(float) + 0.5 * (x >= 0.7).astype(float)

    if 1: 
        plt.plot(x, y_exp, label='y_exp')
        plt.plot(x, y_exp_halfx, label='y_exp_halfx')
        plt.plot(x, y_exp_2x_minus_half, label='y_exp_2x_minus_half')
        plt.plot(x, y_onestep, label='y_onestep')
        plt.plot(x, y_twostep, label='y_twostep')
        plt.legend()
        plt.grid()
        plt.show()

    #scores 
    if 1:
    # Prepare test curves
        test_curves = {
            "y_exp": y_exp,
            "y_exp_halfx": y_exp_halfx,
            "y_exp_2x_minus_half": y_exp_2x_minus_half,
            "y_onestep": y_onestep,
            "y_twostep": y_twostep,
        }
        # Convert each curve to 2D array shape (n_samples, n_budgets)
        # Here we treat each as a single sample with 101 "budgets"
        results = {}
        for name, y in test_curves.items():
            y_mat = y[np.newaxis, :]
            # 1. Breakthroughness legacy=True (default diff_average='median_squared')
            legacy_score = breakthrough_score(x=x, y=y_mat, legacy=True)
            # 2. Breakthroughness legacy=False with diff_average="mean_sqrt"
            breakthrough_plus = breakthrough_score(x=x, y=y_mat, legacy=False, diff_average="mean_sqrt")
            # 3. Skewness with magnitude_weight=True and False
            skew_mag_weight = differences_skew_score(y_mat, magnitude_weight=True)
            skew_no_weight = differences_skew_score(y_mat, magnitude_weight=False)
            results[name] = {
                "breakthroughness_legacy": legacy_score[0],
                "breakthroughness_meansqrt": breakthrough_plus[0],
                "skewness_magweight": skew_mag_weight[0],
                "skewness_noweight": skew_no_weight[0],
            }
        # Print results
        import pandas as pd
        import matplotlib.pyplot as plt

        # Build a DataFrame: functions as rows, scores as columns
        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = "Function"
        df.columns = [
            "Breakthroughness (legacy)",
            "Breakthroughness (mean_sqrt)",
            "Skewness (mag_weighted)",
            "Skewness (not_weighted)"
        ]
        # Reorder columns for clarity
        df = df[[
            "Breakthroughness (legacy)",
            "Breakthroughness (mean_sqrt)",
            "Skewness (mag_weighted)",
            "Skewness (not_weighted)"
        ]]
        
        print("==== Breakthroughness and Skewness Scores Table ====")

        # Just display the DataFrame as a pretty table (pd.DataFrame output)
        print(df.round(3).to_string())


if 0: 
    import numpy as np 
    import pandas as pd
    from analysis.core import breakthrough_score, differences_skew_score 

    # Define the arrays
    y1 = np.array([0,0,0,0,1.0])
    y2 = np.array([0,0,0,0,0.8])
    y3 = np.array([0,0,0,0,0.6])

    x = np.arange(len(y1))

    ys = {
        "jump to 1.0": y1,
        "jump to 0.8": y2,
        "jump to 0.6": y3
    }

    results = {}
    for name, y in ys.items():
        # 1. breakthroughness with legacy=True
        score_legacy = breakthrough_score(x=x, y=y[np.newaxis, :], legacy=True)[0]
        # 2. breakthroughness with diff_average="mean_sqrt", square_numerator=False
        score_meansqrt = breakthrough_score(x=x, y=y[np.newaxis, :], legacy=False, diff_average="mean_sqrt", square_numerator=False)[0]
        # 3. breakthroughness with diff_average="mean_sqrt", square_numerator=True
        score_meansqrt_sqnum = breakthrough_score(x=x, y=y[np.newaxis, :], legacy=False, diff_average="mean_sqrt", square_numerator=True)[0]
        results[name] = {
            "Breakthroughness (legacy)": score_legacy,
            "Breakthroughness (mean_sqrt)": score_meansqrt,
            "Breakthroughness (mean_sqrt, square_numerator=True)": score_meansqrt_sqnum
        }

    df = pd.DataFrame.from_dict(results, orient="index")
    print("==== Step function breakthrough scores table ====")
    print(df.round(5).to_string())

    breakpoint()


    
if 0:
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0,1,101)
    np.random.seed(42)
    # Scale noise with sharpness of curve
    noise1 = np.random.normal(0, 0.05, len(x))   # linear: less sharp
    noise2 = np.random.normal(0, 0.10, len(x))   # quadratic: medium sharp
    noise3 = np.random.normal(0, 0.10, len(x))   # 8th order: very sharp

    y1 = x + noise1
    diffs1 = np.diff(y1)
    sorted_diffs1 = np.sort(diffs1)
    clipped_diffs1 = np.clip(sorted_diffs1, 0, None)
    cumulative_diffs1 = np.cumsum(clipped_diffs1)
    delta_y_total1 = np.sum(diffs1)

    y2 = x**2 + noise2
    diffs2 = np.diff(y2)
    sorted_diffs2 = np.sort(diffs2)
    clipped_diffs2 = np.clip(sorted_diffs2, 0, None)
    cumulative_diffs2 = np.cumsum(clipped_diffs2)
    delta_y_total2 = np.sum(diffs2)

    # Step function: 0 everywhere, goes to 1 at x >= 0.7
    y3 = (x >= 0.7).astype(float)
    diffs3 = np.diff(y3)
    sorted_diffs3 = np.sort(diffs3)
    clipped_diffs3 = np.clip(sorted_diffs3, 0, None)
    cumulative_diffs3 = np.cumsum(clipped_diffs3)
    delta_y_total3 = np.sum(diffs3)

    if 0: 
        plt.plot(x, y1, label='y1 (linear)')
        plt.plot(x, y2, label='y2 (quadratic)')
        plt.plot(x, y3, label='y3 (8th order)')
        plt.legend()
        plt.grid()
        plt.show()

    #sorted diffs
    if 1:
        plt.plot(sorted_diffs1, label='diffs1 (linear)')
        plt.plot(sorted_diffs2, label='diffs2 (quadratic)')
        plt.plot(sorted_diffs3, label='diffs3 (8th order)')
        plt.title('Sorted diffs')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

    if 1:
        plt.plot(clipped_diffs1, label='diffs1 (linear)')
        plt.plot(clipped_diffs2, label='diffs2 (quadratic)')
        plt.plot(clipped_diffs3, label='diffs3 (8th order)')
        plt.title('Clipped diffs')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

    if 1:
        plt.plot(cumulative_diffs1/delta_y_total1, label='diffs1 (linear)')
        plt.plot(cumulative_diffs2/delta_y_total2, label='diffs2 (quadratic)')
        plt.plot(cumulative_diffs3/delta_y_total3, label='diffs3 (8th order)')
        plt.title('Cumulative diffs')
        plt.xlabel('Cumulative index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

    #unsorted diffs
    if 0:
        plt.plot(diffs1, label='diffs1 (linear)')
        plt.plot(diffs2, label='diffs2 (quadratic)')
        plt.plot(diffs3, label='diffs3 (8th order)')
        plt.title('Unsorted diffs')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()
    


