from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import pickle
import subprocess
import os
import shutil
import uuid
import atexit

# starting the app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# location for the empty folder and to upload pdfs
UPLOAD_FOLDER = './uploaded_files/pdf'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
folder_to_empty = './uploaded_files/pdf'
if not os.path.exists(folder_to_empty):
    os.makedirs(folder_to_empty)


def empty_folder(folder_path):
    """
    Function to delete all files in a folder.
    :param folder_path: path to folder
    :return: Error message in case of an error
    """
    try:
        if os.path.exists(folder_path):
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

            return f"Folder '{folder_path}' has been emptied."
        else:
            return f"Folder '{folder_path}' does not exist."
    except Exception as e:
        return f"An error occurred: {str(e)}"


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# setting the variables as global and initiate with None
filtered_df = None
try:
    pickle_file = 'path_to_files'
    df = pd.read_pickle(pickle_file)
except:
    df = pd.DataFrame([])
    print('No data in the backend. Please specify path to files')
filtered_df1 = None
df_current = None
new_data = False


def get_user_folder():
    """
    Creates a user id and a folder for the user if they don't exist.
    :return: the user folder
    """
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())  # Generate a unique identifier for the session
    user_folder = os.path.join(UPLOAD_FOLDER, session['user_id']) + '/'
    os.makedirs(user_folder, exist_ok=True)
    print('user_folder', user_folder)
    return user_folder


@app.route('/')
def home():
    """
    Reqeust to home page.
    :return: tool home page
    """
    return render_template('tool.html')


@app.route('/loading_uploaded')
def loading_uploaded():
    """
    Reqeust to loading page only uploaded.
    :return: loading page
    """
    return render_template('loading_uploaded.html')


@app.route('/loading_database_and_uploaded')
def loading_database_and_uploaded():
    """
    Reqeust to loading page for database and uplaoded.
    :return: loading page
    """
    return render_template('loading_database_and_uploaded.html')


def render_tool2(df_show, warning=None):
    """
    Render tool page with interactive pie chart and table.
    :param df_show: datfarme to display
    :param warning: possible warning message
    :return: tool page rendered with interactive pie chart
    """
    value_counts = df_show['topic'].value_counts().to_frame()  # getting the values of the column One_label
    names = value_counts.index.tolist()
    counts = value_counts['count'].values.tolist()
    max_label = names[0]
    return render_template('tool2.html', labels=names, values=counts,
                           max_label=max_label, warning_message=warning)  # passing to values to tool2.html


@app.route('/show_pie_chart_database', methods=['GET'])
def show_pie_chart_database():
    """
    Render pie chart with existing dataframe.
    :return: pie chart page with existing dataframe
    """
    global df
    global df_current
    df_current = df.copy()
    return render_tool2(df_current)


@app.route('/process_uploaded_data', methods=['GET', 'POST'])
def process_uploaded_data():
    """
    Preprocess uploaded data.
    :return: 'dataframe created' if successful
    """
    global new_data
    global df_current
    user_path = './5_Tool/uploaded_files/df_uploaded_' + session['user_id'] + '.pckl'
    if (not os.path.isfile(user_path)) or new_data:
        script_path = './5_Tool/preprocessing.py'
        process = subprocess.Popen(['python3', script_path, session['user_id']], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stderr)
        print(stdout)
        return_code = process.returncode

    if not os.path.isfile(user_path):
        raise (Exception)
    new_data = False
    with open(user_path, 'rb') as f:
        df_current = pickle.load(f)
    return "Dataframe created"


@app.route('/error1', methods=['GET', 'POST'])
def error1():
    """
    Render error in preprocesing.
    :return: Error empty dataframe
    """
    print('error1')
    warning_message = "Warning: The data you uploaded could not be processed or does not contain any relevant information."
    return render_template('empty_dataframe.html', warning_message=warning_message)


@app.route('/error2', methods=['GET', 'POST'])
def error2():
    """
    Render error in preprocessing but still displaying the existing data.
    :return: Pie chart and error
    """
    print('error2')
    global df
    global df_current
    warning = 'The data you uploaded could not be processed or does not contain any relevant information. Only showing the data in the backend instead.'
    df_current = df.copy()
    return render_tool2(df_current, warning)


@app.route('/show_pie_chart_uploaded', methods=['GET', 'POST'])
def show_pie_chart_uploaded():
    """
    Render pie chart with uploaded data.
    :return: pie chart page with uploaded data
    """
    try:
        global df_current
        return render_tool2(df_current)
    except Exception as e:
        warning_message = "Warning: The data you uploaded could not be processed or does not contain any relevant information."
        print(e)
        return render_template('empty_dataframe.html', warning_message=warning_message)


@app.route('/show_pie_chart_database_and_uploaded', methods=['GET'])
def show_pie_chart_database_and_uploaded():
    """
    Render pie chart with existing and uploaded data.
    :return: pie chart page with existing uploaded data
    """
    global df
    global df_current
    try:
        df_current = pd.concat([df, df_current], ignore_index=True)  # concatenate the both df
        return render_tool2(df_current)
    except Exception as e:
        warning = 'The data you uploaded could not be processed or does not contain any relevant information. Only showing the data in the backend instead.'
        df_current = df.copy()
        return render_tool2(df_current, warning)


@app.route('/update_table', methods=['GET', 'POST'])
def update_table():
    """
    Updates the table when a topic is selected and/or something is searched.
    :return: updated html dictionary
    """
    global filtered_df
    global df_current
    global filtered_df1

    data = request.get_json()
    label = data.get('label')
    searched = data.get('searched')
    new_search = data.get('new_search')

    if label and not searched:
        # Handle the case when only a topic is selected
        index = df_current['topic'] == label
        filtered_df1 = df_current.loc[index]
        filtered_df = filtered_df1.copy()
        text_column = filtered_df['text']

        html_table = '<table border="1" cellspacing="0" cellpadding="5"><tr></tr>'
        for row in text_column:
            html_table += f'<tr><td>{row}</td></tr>'
        html_table += '</table>'

        return jsonify({'html': html_table, 'results_found': True})
    # elif topic selected and searched:
    else:
        # Handle the case when both topic selected and searched are present
        filtered_df = df_current[df_current['text'].str.contains(searched, case=False)]

        if filtered_df is None or filtered_df.empty:
            return jsonify({'html': 'No results found.', 'labels': None, 'values': None, 'max_label': None,
                            'results_found': False})

        value_counts = filtered_df['topic'].value_counts().to_frame()  # getting the values of the column One_label
        names = value_counts.index.tolist()
        counts = value_counts['count'].values.tolist()
        max_label = names[0]

        label_to_display = max_label if new_search else label
        index = filtered_df['topic'] == label_to_display
        filtered_df_topic = filtered_df.loc[index]

        if filtered_df_topic is None or filtered_df_topic.empty:
            return jsonify({'html': 'No results found.', 'labels': None, 'values': None, 'max_label': None,
                            'results_found': False})

        text_column = filtered_df_topic['text']

        html_table = '<table border="1" cellspacing="0" cellpadding="5"><tr></tr>'
        for row in text_column:
            html_table += f'<tr><td>{row}</td></tr>'
        html_table += '</table>'

        return jsonify(
            {'html': html_table, 'labels': names, 'values': counts, 'max_label': max_label, 'results_found': True})


@app.route('/upload_files', methods=['POST'])
def upload_files():
    """
    Upload the selected pdf files into the defined folder.
    :return: status message
    """
    global new_data
    user_folder = get_user_folder()
    if request.method == 'POST':
        files = request.files.getlist('files')
        if files:
            for file in files:
                filename = file.filename
                file.save(os.path.join(user_folder, filename))
            new_data = True
            return "Files uploaded successfully!"
    return "No files to upload."


@app.route('/reset_data', methods=['POST'])
def reset_data():
    """
    Reset the files by deleting all uploaded and selected files.
    :return: status message
    """
    try:
        # Empty UPLOAD_FOLDER
        user_folder = get_user_folder()
        empty_folder(user_folder)
        # Remove or reset the pickle file
        if os.path.exists('./5_Tool/uploaded_files/df_uploaded_' + session['user_id'] + '.pckl'):
            os.remove('./5_Tool/uploaded_files/df_uploaded_' + session['user_id'] + '.pckl')

        return jsonify({'message': 'All data has been reset successfully.'})
    except Exception as e:
        return jsonify({'message': f'Error occurred: {str(e)}'}), 500


@app.route('/list_uploaded_files')
def list_uploaded_files():
    """
    List all uploaded filed.
    :return: file list as json
    """
    user_folder = get_user_folder()
    files = os.listdir(user_folder)
    return jsonify(files)


@app.route('/remove_file', methods=['POST'])
def remove_file():
    """
    Removes a selected file.
    :return: status message
    """
    data = request.get_json()
    file_name = data['fileName']
    user_folder = get_user_folder()
    file_path = os.path.join(user_folder, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'message': 'File removed successfully'})
    return jsonify({'message': 'File not found'}), 404


@app.route('/remove_uploaded_file', methods=['POST'])
def remove_uploaded_file():
    """
    Removes an uploaded file.
    :return: status message
    """
    global new_data
    data = request.get_json()
    file_name = data['fileName']
    user_folder = get_user_folder()
    file_path = os.path.join(user_folder, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        new_data = True
        # Remove or reset the pickle file
        if os.path.exists('./5_Tool/uploaded_files/df_uploaded_' + session['user_id'] + '.pckl'):
            os.remove('./5_Tool/uploaded_files/df_uploaded_' + session['user_id'] + '.pckl')
        return jsonify({'message': 'File removed successfully'})
    return jsonify({'message': 'File not found'}), 404


def cleanup():
    """
    Deletes all session data.
    """
    shutil.rmtree('./5_Tool/uploaded_files/pdf/')
    for filename in os.listdir('./5_Tool/uploaded_files/'):
        if filename.startswith('df_uploaded_'):
            file_path = os.path.join('./5_Tool/uploaded_files/', filename)
            os.remove(file_path)


# Register the cleanup function with atexit
atexit.register(cleanup)

if __name__ == '__main__':
    # Start the app
    print(app.root_path)
    app.run(debug=False, host='0.0.0.0', port=8000)
