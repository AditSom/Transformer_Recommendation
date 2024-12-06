import numpy as np
import matplotlib.pyplot as plt    
import matplotlib.colors as mcolors

def plot(splitdata, usridx, model, gender='Man', save=False):
    """
    Plot split results for a (user, model, gender) tuple
    Inputs:
        splitdata: dict - The data structure containing split information.
        usridx: int - The user index to extract data for.
        model: list - model to plot.
        gender: str - The gender of the user.
        save: bool - Whether to save the plot to a file.
    """
    liked_acc_split = []
    disliked_acc_split = []
    liked_var_split = []
    disliked_var_split = []
    modeldata = splitdata[model]
    splits = list(modeldata.keys())
    for split in splits:
        # print(split)
        usrdata = (modeldata[split][gender][usridx])
        liked_acc_split.append(usrdata["liked_acc"])
        disliked_acc_split.append(usrdata["disliked_acc"])
        liked_var_split.append(usrdata["liked_var"])
        disliked_var_split.append(usrdata["disliked_var"])

    lv = np.sqrt(liked_var_split)
    dv = np.sqrt(disliked_var_split)
    plt.plot(splits, liked_acc_split, color='orange')
    plt.plot(splits, disliked_acc_split, color='#1f77b4')
    plt.errorbar(splits, liked_acc_split, yerr=lv, fmt='o', capsize=6, ecolor='dimgray',
                 mfc='orange' ,mec='white', label='liked')
    plt.errorbar(splits, disliked_acc_split, yerr=dv, fmt='o', capsize=6, ecolor='darkgray',
                 mfc='#1f77b4',mec='white', label='disliked')	

    # plt.legend(["liked", 'disliked'])
    plt.legend()
    plt.xlabel("split")
    plt.ylabel("accuracy")
    title = f"cross_validation for {model} user_{usridx}_{gender}"
    plt.title(title)
    if(save):
        plt.savefig(title+'.pdf')
    plt.show()
    
    
def plot_all_models(splitdata, usridx, models, gender='Man', save=False):
    """
    Plots accuracy and error bars for all specified models in subplots.

    Inputs:
        splitdata: dict - The data structure containing split information.
        usridx: int - The user index to extract data for.
        models: list - List of model names to plot.
        gender: str - The gender of the user.
        save: bool - Whether to save the plot to a file.
    """
    # Determine the layout of subplots
    n_models = len(models)
    n_cols = 3  # Define how many subplots per row
    n_rows = (n_models + n_cols - 1) // n_cols  # Calculate rows based on number of models

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to iterate easily

    for idx, model in enumerate(models):
        liked_acc_split = []
        disliked_acc_split = []
        liked_var_split = []
        disliked_var_split = []

        modeldata = splitdata[model]
        splits = list(modeldata.keys())

        for split in splits:
            usrdata = (modeldata[split][gender][usridx])
            liked_acc_split.append(usrdata["liked_acc"])
            disliked_acc_split.append(usrdata["disliked_acc"])
            liked_var_split.append(usrdata["liked_var"])
            disliked_var_split.append(usrdata["disliked_var"])

        lv = np.sqrt(liked_var_split)
        dv = np.sqrt(disliked_var_split)
        # lv = (liked_var_split)
        # dv = (disliked_var_split)

        # Plot on the corresponding subplot
        ax = axes[idx]
        ax.plot(splits, liked_acc_split, color='orange', label='liked (mean)')
        ax.plot(splits, disliked_acc_split, color='#1f77b4', label='disliked (mean)')
        ax.errorbar(splits, liked_acc_split, yerr=lv, fmt='o', capsize=6, ecolor='#D2B48C',
                    mfc='orange', mec='white', label='liked (std)')
        ax.errorbar(splits, disliked_acc_split, yerr=dv, fmt='o', capsize=6, ecolor='darkgray',
                    mfc='#1f77b4', mec='white', label='disliked (std)')

        ax.set_xlabel("split")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{model} | user {usridx} | {str(gender).lower()}")
        ax.legend()
        ax.grid(True)

    # Remove unused axes (if models < n_rows * n_cols)
    for ax in axes[len(models):]:
        ax.axis('off')

    # Adjust layout and optionally save
    plt.tight_layout()
    if save:
        plt.savefig(f"cross_validation_user_{usridx}_{gender}.pdf")
    plt.show()


def plot_models_avgstats(splitdata_avg, gender="Man" ,save=False):
    """
    Plots accuracy and error bars for all specified models in subplots.

    Parameters:
        splitdata: dict - The data structure containing split information.
        usridx: int - The user index to extract data for.
        models: list - List of model names to plot.
        gender: str - The gender of the user.
        save: bool - Whether to save the plot to a file.
    """
    # Determine the layout of subplots
    models = list(splitdata_avg.keys())
    n_models = len(models)
    n_cols = 3  # Define how many subplots per row
    n_rows = (n_models + n_cols - 1) // n_cols  # Calculate rows based on number of models

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to iterate easily

    for idx, model in enumerate(models):
        liked_acc_split = []
        disliked_acc_split = []
        liked_var_split = []
        disliked_var_split = []

        modeldata = splitdata_avg[model]
        splits = list(modeldata.keys())

        for split in splits:
            usrdata = (modeldata[split][gender])
            # print(usrdata)
            liked_acc_split.append(usrdata["liked_acc"])
            disliked_acc_split.append(usrdata["disliked_acc"])
            liked_var_split.append(usrdata["liked_var"])
            disliked_var_split.append(usrdata["disliked_var"])

        lv = np.sqrt(liked_var_split)
        dv = np.sqrt(disliked_var_split)
        # lv = (liked_var_split)
        # dv = (disliked_var_split)

        # Plot on the corresponding subplot
        ax = axes[idx]
        ax.plot(splits, liked_acc_split, color='orange', label='liked (mean)')
        ax.plot(splits, disliked_acc_split, color='#1f77b4', label='disliked (mean)')
        ax.errorbar(splits, liked_acc_split, yerr=lv, fmt='o', capsize=6, ecolor='#D2B48C',
                    mfc='orange', mec='white', label='liked (std)')
        ax.errorbar(splits, disliked_acc_split, yerr=dv, fmt='o', capsize=6, ecolor='darkgray',
                    mfc='#1f77b4', mec='white', label='disliked (std)')

        ax.set_xlabel("split")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{model} | {str(gender).lower()}")
        ax.legend(prop={'size': LSIZE})
        ax.grid(True)

    # Remove unused axes (if models < n_rows * n_cols)
    for ax in axes[len(models):]:
        ax.axis('off')

    # Adjust layout and optionally save
    plt.tight_layout()
    if save:
        plt.savefig(f"./train_splits_{gender}.pdf")
    plt.show()



def darken_color(color, factor=1):
    """
    Darkens the given color by multiplying (1-factor) to its RGB values.
    :param color: Color name or hex code (e.g., 'orange' or '#1f77b4')
    :param factor: Factor by which to darken the color (0 means no change, 1 means black)
    :return: Darkened color as a hex code
    """
    rgb = mcolors.to_rgb(color)  # Convert color to RGB
    dark_rgb = [c * factor for c in rgb]  # Apply the darkening factor
    return mcolors.to_hex(dark_rgb)  # Convert back to hex

# Original colors
orange = 'orange'
blue = '#1f77b4'

# Darkened colors
dark_orange = darken_color(orange)
dark_blue = darken_color(blue)

def plot_models_avgstats_bar(splitdata_avg, gender="Man", liked=True, save=False):
    """
    Plots accuracy and error bars for all specified models in subplots.

    Parameters:
        splitdata: dict - The data structure containing split information.
        usridx: int - The user index to extract data for.
        models: list - List of model names to plot.
        gender: str - The gender of the user.
        save: bool - Whether to save the plot to a file.
    """
    # Determine the layout of subplots
    models = list(splitdata_avg.keys())
    n_models = len(models)

    fig = plt.figure(figsize=(12, 6))
    plt.xlabel("model")
    plt.ylabel("accuracy")
    xrange = np.arange(0, len(models))
    liked_accs = []
    disliked_accs = []
    liked_vars = []
    disliked_vars = []


    for idx, model in enumerate(models):
        modeldata = splitdata_avg[model][0.9][gender]
        liked_accs.append(modeldata["liked_acc"])
        disliked_accs.append(modeldata["disliked_acc"])
        liked_vars.append(modeldata["liked_var"])
        disliked_vars.append(modeldata["disliked_var"]) 

    bwid = 0.3
    plt.bar(xrange-bwid/2, liked_accs, yerr=liked_vars, color=orange, alpha=1, width=bwid,
             label='liked (mean)',  capsize=5, ecolor='#708090')
    plt.bar(xrange+bwid/2, disliked_accs, yerr=disliked_vars, color=blue, alpha=1, width=bwid,
             label='disliked (mean)',  capsize=5, ecolor='#FF7F50')

    # plt.errorbar(xrange, liked_vars, color=dark_orange, alpha=1, linewidth=2, label='liked (var)')
    # plt.errorbar(xrange, disliked_vars, color=dark_blue, alpha=1, linewidth=2, label='disliked (var)')

    plt.xticks(xrange, models, rotation=45, fontsize=LSIZE)
    plt.ylim(0, 1)
    plt.title(f"Accuracy at split = 0.9 for {gender.lower()} profiles")
    plt.legend(fontsize=LSIZE)
    plt.tight_layout()
    if save:
        plt.savefig(f"./split_9_allmodels_{gender.lower()}.pdf")
    plt.show()
        
save = True
plot_models_avgstats_bar(splitdata_avg, save=save)
plot_models_avgstats_bar(splitdata_avg, save=save, gender="Woman")