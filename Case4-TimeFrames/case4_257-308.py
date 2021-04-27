import os
import mind_reading as mr
import pandas as pd

path = '/home/data/NDClab/lab_projects/mind_reading/data_resource/'
output_path = '/home/data/NDClab/lab_projects/mind_reading/scripts/output/'

# list all folders' name
participants = os.listdir(path)

# select time windows
time_window1 = 257
time_window2 = 308

# Create folder for accuracy inside of output
if not os.path.exists(output_path + f'{time_window1}_{time_window2}'):
    os.mkdir(output_path + f'{time_window1}_{time_window2}')

# remove the 'cha' folder we don't need
# participants = participants.remove('cha')

# create the initial dataframe
df_accr = pd.DataFrame(index=['SVC', 'DTC', 'NB'])
df_preci = pd.DataFrame(index=['SVC', 'DTC', 'NB'])
df_cross_val = pd.DataFrame(index=['SVC', 'DTC', 'NB'])

for participant in participants:
    # iterate all the folders

    for file in os.listdir(path + participant):

        # iterate all files in every folder, find out the one end with 'Cong.csv' and 'Incong.csv' as input data
        if file.endswith('Cong.csv'):
            file1 = f"{path}{participant}/{file}"
        if file.endswith('Incong.csv'):
            file2 = f"{path}{participant}/{file}"

    # load in cong and incong data for them
    df1 = mr.load_data(file1)
    df2 = mr.load_data(file2)

    # concatenate such data
    data = mr.concatenate_data(df1, df2)

    # find trials to later separate
    trials_index = mr.find_trials(data)

    # separate trials
    trials = mr.separate_trials(data, trials_index)

    # create the label column
    labels = mr.create_multi_labels(data)

    # Go through each trial, reset the columns, we split from 100-300ms ((308th sample to 513th sample))
    pro_trials = mr.process_trials(trials, time_window1, time_window2)

    # Find the mean across channels
    avg_trials = mr.average_trials(pro_trials)

    # concatenates the average trials dataframe with labels
    ml_df = mr.create_ml_df(avg_trials, labels)

    # train models
    X_train, X_test, y_train, y_test = mr.prepare_ml_df(ml_df)

    acc_svc, precision_svc, acc_svc_cv, pred_svc = mr.train_svc_multi(
        X_train, X_test, y_train, y_test)

    acc_dtc, precision_dtc, acc_dtc_cv, pred_dtc = mr.train_dtc_multi(
        X_train, X_test, y_train, y_test)

    acc_nb, precision_nb, acc_nb_cv, pred_nb = mr.train_nb_multi(
        X_train, X_test, y_train, y_test)

    # Save confusion matrix for each model
    participant_path = f'{path}{participant}' + '/'

    mr.save_confusion_matrix(participant_path, y_test,
                             pred_svc, labels, participant, 'svc', time_window1, time_window2)
    mr.save_confusion_matrix(participant_path, y_test,
                             pred_dtc, labels, participant, 'dtc', time_window1, time_window2)
    mr.save_confusion_matrix(participant_path, y_test,
                             pred_nb,  labels, participant, 'nb', time_window1, time_window2)

    # acc_nn, precision_nn = mr.train_nn_multi(
    #     64, X_train, X_test, y_train, y_test)

    # add every participant's accuracy together
    acc_list = [f"{acc_svc:.2f}", f"{acc_dtc:.2f}", f"{acc_nb:.2f}"]
    prec_list = [f"{precision_svc:.2f}",
                 f"{precision_dtc:.2f}", f"{precision_nb:.2f}"]
    acc_cv_list = [f"{acc_svc_cv:.2f}",
                   f"{acc_dtc_cv:.2f}", f"{acc_nb_cv:.2f}"]
    # , f"{precision_nn:.2f}"]
    # , f"{acc_nn:.2f}"]

    df_accr = mr.res_df(df_accr, acc_list, participant)
    df_preci = mr.res_df(df_preci, prec_list, participant)
    df_cross_val = mr.res_df(df_cross_val, acc_cv_list, participant)

# generate result .csv file
df_accr.to_csv(
    output_path + f'{time_window1}_{time_window2}/case_4_accuracy_{time_window1}-{time_window2}.csv')

df_preci.to_csv(
    output_path + f'{time_window1}_{time_window2}/case_4_precision_{time_window1}-{time_window2}.csv')

df_cross_val.to_csv(
    output_path + f'{time_window1}_{time_window2}/case_4_cv_acc_{time_window1}-{time_window2}.csv')
