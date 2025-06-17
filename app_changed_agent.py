# -*- coding: utf-8 -*-

import traceback

from shapely.geometry import Polygon, mapping

from flask import Flask, Response, stream_with_context, request, render_template, jsonify, session, redirect, url_for

from openai import OpenAI
import os
from dotenv import load_dotenv
import ast
from user_agents import parse
import requests

# import logging
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()
import sys
from io import StringIO
from datetime import datetime
from flask_socketio import SocketIO, emit
from ask_functions_agent import *
from agent_search_fast import id_list_of_entity_fast

# from repair_data import repair,crs_transfer
# from shape2ttf import shapefile_to_ttl
output = StringIO()
original_stdout = sys.stdout
app = Flask(__name__)
# app.logger.setLevel(logging.WARNING)

app.secret_key = 'secret_key'  # 用于启用 flash() 方法发送消息
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.before_request
def initialize_session():
    if 'globals_dict' not in session:
        session['globals_dict'] = {'bounding_box_region_name': 'Munich',
                                   'bounding_coordinates': [48.061625, 48.248098, 11.360777, 11.72291],
                                   'bounding_wkb': '01030000000100000005000000494C50C3B7B82640D9CEF753E3074840494C50C3B7B82640FC19DEACC11F484019E76F4221722740FC19DEACC11F484019E76F4221722740D9CEF753E3074840494C50C3B7B82640D9CEF753E3074840'}


        print('session 初始化')


geo_functions.global_id_attribute = {}
geo_functions.global_id_geo = {}

# global_variables = {}

# 示例的 Markdown 文本（包含图片链接）
# ![示例图片](/static/data.png "这是一个示例图片")
socketio = SocketIO(app, manage_session=True, async_mode='threading')


# async def async_task(news):
#     # 你的异步代码
#     print(search.run(news))
#
# def search_internet(news):
#     # 使用 eventlet 来 "模拟" 异步操作
#     pool = eventlet.GreenPool()
#     pool.spawn(async_task,news)
#     pool.waitall()


@socketio.on('join')
def on_join(data):
    session['sid'] = request.sid
    send_data(session['sid'], 'sid', sid=session['sid'])

    # print(session['username'])


geojson_files = {
    '1': 'buildings_geojson.geojson',
    '2': 'land_geojson.geojson',
    '3': 'soil_maxvorstadt_geojson.geojson',
    '4': 'points_geojson.geojson',
    '5': 'lines_geojson.geojson',
}
geojson_data = {}

for key, filepath in geojson_files.items():
    with open('static/geojson' + '/' + filepath, 'r', encoding='utf-8') as file:
        geojson_data[key] = json.load(file)


@app.route('/introduction')
def introduction():
    return render_template('introduction.html')


@app.route('/geojson/<key>')
def send_geojson(key):
    return jsonify(geojson_data.get(key, {}))


@app.route('/submit-qu', methods=['POST'])
def submit_qu():
    start_time = request.form.get('start_time')
    end_time = time.time()
    ip_address = request.remote_addr
    answers = {key: value for key, value in request.form.items() if key != 'start_time'}

    response_data = {
        'ip_address': ip_address,
        'start_time': start_time,
        'end_time': end_time,
        'answers': answers
    }

    with open('responses.jsonl', 'a') as f:
        f.write(json.dumps(response_data) + '\n')

    return redirect(url_for('thank_you'))


@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')


@app.route('/question')
def question():
    return render_template('question.html')


def complete_json(input_stream):
    """
    尝试补全一个不完整的JSON字符串，包括双引号和括号。

    参数:
    - input_stream: 一个可能不完整的JSON字符串

    返回:
    - 完整的JSON对象
    """
    brackets = {'{': '}', '[': ']'}
    quotes = {'"': '"'}
    stack = []
    in_string = False

    # 尝试解析JSON，如果失败则补全
    try:
        return json.loads(input_stream)
    except json.JSONDecodeError:
        pass  # 继续到补全逻辑

    # 逐字符遍历输入字符串
    for char in input_stream:
        if char in quotes and not in_string:
            # 进入字符串
            in_string = True
            stack.append(quotes[char])
        elif char in quotes and in_string:
            # 结束字符串
            in_string = False
            stack.pop()
        elif char in brackets and not in_string:
            # 进入对象或数组
            stack.append(brackets[char])
        elif stack and char == stack[-1] and not in_string:
            # 退出对象或数组
            stack.pop()

    # 补全字符串
    if in_string:
        input_stream += '"'
        stack.pop()

    # 根据栈中剩余的括号补全JSON
    while stack:
        input_stream += stack.pop()

    # 再次尝试解析补全后的JSON
    return json.loads(input_stream)


# html_content = markdown2.markdown(markdown_text)
def reorder_relations(relations):
    for relation in relations:
        if 0 not in (relation['head'], relation['tail']):
            break
    else:
        # 0 simultaneously exists in all dictionaries
        return relations

    reordered_relations = []
    zero_relation = None
    for relation in relations:
        if 0 in (relation['head'], relation['tail']):
            zero_relation = relation
        else:
            reordered_relations.append(relation)

    if zero_relation:
        reordered_relations.append(zero_relation)

    return reordered_relations


def extract_code_blocks(code_str):
    code_blocks = []
    parts = code_str.split("```python")
    for part in parts[1:]:  # 跳过第一个部分，因为它在第一个代码块之前
        code_block = part.split("```")[0]
        code_blocks.append(code_block)
    return code_blocks


def len_str2list(result):
    if result == None:
        return 0
    # result=result.replace("None","").replace(" ","")
    try:
        # 尝试使用 ast.literal_eval 解析字符串
        result = ast.literal_eval(result)
        # 检查解析结果是否为列表
        if result != None:
            if not isinstance(result, int):
                return len(result)
            else:
                return result
        else:
            return ''
    except (ValueError, SyntaxError):
        # 如果解析时发生错误，说明字符串不是有效的列表字符串
        try:
            dict_result = json.loads(result)
            return len(dict_result)
        except:

            return str(len(result)) + "(String)"


def judge_list(result):
    try:
        # 尝试使用 ast.literal_eval 解析字符串
        result = ast.literal_eval(result)
        # 检查解析结果是否为列表
        return True
    except (ValueError, SyntaxError):
        # 如果解析时发生错误，说明字符串不是有效的列表字符串
        return False


def details_span(result, run_time):
    if result == None:
        result = "None"
    pattern = r"An error occurred:.*\)"
    match = re.search(pattern, result)

    if match:
        error_message = match.group(0)
        aa = error_message
        return {'error': result + " \n " + aa}
    else:
        length = len_str2list(str(result))
        attention = ''
        if 'String' not in str(length) and int(length) > 10000:
            attention = 'Due to the large volume of data, visualization may take longer.'
            if int(length) == 20000:
                attention = 'Due to the large volume of data in your current search area, only 20,000 entries are displayed.'
        aa = f"""
\n
<details>
    <summary>`Code result: Length:{length},Run_time:{round(run_time, 2)}s`</summary>
       {str(result)}
</details>
{attention}


            """

        return {'normal': aa}


def short_response(text_list):
    if len(str(text_list)) > 1000 and judge_list(text_list):
        if len_str2list(text_list) < 40:
            result = ast.literal_eval(text_list)
            short_suitable_types = [t[:35] for t in result]
            return str(short_suitable_types)
        else:
            return text_list[:600]
    else:
        return text_list[:600]


# 设置文件上传的目录和大小限制
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'doc', 'docx', 'xlsx', 'csv', 'ttl'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         session['uploaded_indication'] = filename
#         return jsonify({"filename": filename}), 200
#     else:
#         return jsonify({"error": "File type not allowed"}), 400


@app.route('/debug_mode', methods=['POST'])
def debug_mode():
    data = request.get_json().get('message')  # 获取JSON数据
    if data == 'debug':
        session['template'] = True
    else:
        session['template'] = False
    return jsonify({"text": True}), 400


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "The file is too large. Maximum file size is 50MB."}), 413


@app.route('/')
def home():

    print("initial")
    del_uploaded_sql()

    ip_ = request.remote_addr
    session['file_path']=''
    session['ip_'] = ip_
    session['uploaded_indication'] = None
    session['sid'] = ''

    user_agent_string = request.headers.get('User-Agent')
    if user_agent_string:
        user_agent = parse(user_agent_string)

        os = user_agent.os.family
        browser = user_agent.browser.family
        if user_agent.is_mobile:
            device_type = 'Mobile'
        elif user_agent.is_tablet:
            device_type = 'Tablet'
        elif user_agent.is_pc:
            device_type = 'Desktop'
        else:
            device_type = 'Unknown'
        session['os'] = os
        session['browser'] = browser
        session['device_type'] = device_type
    else:
        session['os'] = None
        session['browser'] = None
        session['device_type'] = None
    session['globals_dict']=None
    session['template'] = False
    session['history'] = []
    return render_template('index.html')


def polygons_to_geojson(polygons_dict):
    """
    将键值都为Polygon对象的字典转换为键值都为GeoJSON的字典。

    :param polygons_dict: 一个键值都为Polygon对象的字典。
    :return: 一个键值都为GeoJSON格式的字典。
    """
    geojson_dict = {}
    for key, polygon in polygons_dict.items():
        # 将每个Polygon对象转换为GeoJSON格式的字典
        geojson_dict[key] = mapping(polygon)
    return geojson_dict


def send_data(data, mode="data", index="", sid=''):
    target_labels = []
    if mode == "map":

        if not isinstance(data, str) and len(data) != 0:
            if 'target_label' in data:
                target_labels = data['target_label']
                data.pop('target_label')
            data = polygons_to_geojson(data)
    if sid != '':
        socketio.emit('text', {mode: data, 'index': index, 'target_label': target_labels}, room=sid)
    else:
        print('no sid')


def find_insert_comment_position(multiline_str, code_line, mode=False):
    lines = multiline_str
    comment_positions = []
    if mode:
        normal_special_char = '#><;'
    else:
        normal_special_char = '#'

    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            comment_positions.append((i, line.strip()))

    for idx, (pos, comment) in enumerate(comment_positions):
        if idx + 1 < len(comment_positions):
            next_comment_pos = comment_positions[idx + 1][0]
            if pos < lines.index(code_line) < next_comment_pos:
                return comment.replace(normal_special_char, '').replace("'", '').strip()
        else:
            if pos < lines.index(code_line):
                return comment.replace(normal_special_char, '').replace("'", '').strip()

    result = (-1, "")
    return result[1][4:].strip()

def print_function(var_name):
    if len(str(var_name))>4000:
        print("What you want to print is too long, it is a %s, its length is %s"%(type(var_name).__name__, len(var_name)))
    else:
        print(var_name)
def process_text_2code(lines, session, sid):

        """
        处理代码字符串，插入 send_data 调用，优化格式，并返回最终的代码字符串。

        参数:
            lines (str): 原始代码字符串
            session (dict): 包含模板信息的会话数据
            sid (str): 会话 ID

        返回:
            str: 处理后的代码字符串
        """
        # 可能包含的函数调用
        lines = lines.split('\n')
        filtered_lst = [item for item in lines if item.strip()]
        lines = filtered_lst
        new_lines = []
        variable_dict = {}
        i = 0

        while i < len(lines):
            each_line = lines[i].strip()
            # Check if this is a multi-line function call
            if '=' in each_line and any(func in each_line for func in
                                        ['geo_filter(', 'id_list_of_entity(','id_list_of_entity_fast(', 'add_or_subtract_entities(',
                                         'area_filter(', 'set_bounding_box(','traffic_navigation(']):
                variable_str = each_line.split('=')[0].strip()
                full_function = each_line
                open_parens = full_function.count('(')
                close_parens = full_function.count(')')
                j = i + 1

                # Collect multi-line function call
                while j < len(lines) and open_parens > close_parens:
                    full_function += '\n' + lines[j]
                    open_parens += lines[j].count('(')
                    close_parens += lines[j].count(')')
                    j += 1

                # Inject bounding_box from session if id_list_of_entity_fast is called
                if 'id_list_of_entity_fast(' in full_function and 'bounding_box=' not in full_function:
                    last_paren_index = full_function.rfind(')')
                    if last_paren_index != -1:
                        # Check if there are existing arguments to decide if a comma is needed
                        open_paren_index = full_function.find('(')
                        content_between_parens = full_function[open_paren_index + 1:last_paren_index].strip()
                        separator = ", " if content_between_parens else ""
                        
                        # Inject the bounding_box parameter
                        full_function = (
                            full_function[:last_paren_index]
                            + f"{separator}bounding_box=session['globals_dict']"
                            + full_function[last_paren_index:]
                        )

                variable_dict[variable_str] = full_function
                comment_index = find_insert_comment_position(lines, lines[i], session['template'])
                new_lines.append(full_function)
                new_lines.append(f"send_data({variable_str}['geo_map'], 'map', '{comment_index}', sid='{sid}')")
                i = j
            elif any(func in each_line for func in
                     ['geo_filter(', 'id_list_of_entity(','id_list_of_entity_fast(', 'area_filter(', 'add_or_subtract_entities(',
                      'set_bounding_box(']) and '=' not in each_line:
                full_function = each_line
                open_parens = full_function.count('(')
                close_parens = full_function.count(')')
                j = i + 1

                # Collect multi-line function call
                while j < len(lines) and open_parens > close_parens:
                    full_function += '\n' + lines[j]
                    open_parens += lines[j].count('(')
                    close_parens += lines[j].count(')')
                    j += 1

                # Inject bounding_box from session if id_list_of_entity_fast is called
                if 'id_list_of_entity_fast(' in full_function and 'bounding_box=' not in full_function:
                    last_paren_index = full_function.rfind(')')
                    if last_paren_index != -1:
                        # Check if there are existing arguments to decide if a comma is needed
                        open_paren_index = full_function.find('(')
                        content_between_parens = full_function[open_paren_index + 1:last_paren_index].strip()
                        separator = ", " if content_between_parens else ""

                        # Inject the bounding_box parameter
                        full_function = (
                            full_function[:last_paren_index]
                            + f"{separator}bounding_box=session['globals_dict']"
                            + full_function[last_paren_index:]
                        )

                comment_index = find_insert_comment_position(lines, lines[i])
                new_lines.append(f"temp_result = {full_function}")
                new_lines.append(f"send_data(temp_result['geo_map'], 'map', '{comment_index}', sid='{sid}')")
                i = j
            else:
                new_lines.append(each_line)
                i += 1

        # Handle the last line if it's a variable or needs print_process
        if new_lines and '=' not in new_lines[-1] and 'send_data' not in new_lines[-1] and 'id_list_explain(' not in \
                new_lines[-1] and '#' not in new_lines[-1]:
            # if new_lines[-1].strip() in variable_dict:
            #     if 'id_list_explain(' in variable_dict[new_lines[-1]]:
            #         pass
            # else:
            new_lines[-1] = f"print_process({new_lines[-1].strip()})"

        code_str = '\n'.join(new_lines)
        return code_str
@app.route('/submit', methods=['POST', 'GET'])
def submit():
    # 加载包含 Markdown 容器的前端页面
    data = request.get_json().get('text')  # 获取JSON数据
    messages = request.get_json().get('messages')  # 获取JSON数据
    sid = request.get_json().get('sid')  # 获取JSON数据
    currentMode = request.get_json().get('currentMode')  # 获取JSON数据
    need_bounding_box_function = ['id_list_of_entity(', 'geo_filter(']
    # new_message = request.get_json().get('new_message')  # 获取JSON数据
    processed_response = []

    def process_code(data):
        yield_list = []
        compelete = False
        template = False
        steps = 0  # 纯文字返回最多两次
        whole_step = 0  # 总返回数,可以调整数值来限制未来的返回数
        true_step = 0  # 总返回数

            # chat_response = str(datetime.now())+"   "+str(len(messages)) #test

        code_list = []
        #
        if session['template'] == True:

            code_list.append(data)
            yield_list.append(data)

        else:

            messages.append(message_template('user', data))
            # print(messages)

            chat_response = (chat_single(messages, "stream"))

            chunk_num = 0
            in_code_block = False
            code_block_start = "```python"
            code_block_end = "```"

            line_buffer = ""
            total_buffer = ""
            total_char_list = []
            for chunk in chat_response:
                if chunk is not None:
                    if chunk.choices[0].delta.content is not None:
                        if chunk_num == 0:
                            char = "\n" + chunk.choices[0].delta.content

                        else:
                            char = chunk.choices[0].delta.content
                        chunk_num += 1
                        total_buffer += char

                        total_char_list.append(char)
                        # print(char, end='', flush=True)

                        line_buffer += char
                        # 检查是否遇到了Python代码块的起始标志
                        if code_block_start.startswith(line_buffer) and not in_code_block:
                            in_code_block = True
                            line_buffer = ""  # 清空行缓冲区
                            continue
                        # 检查是否遇到了Python代码块的结束标志
                        elif code_block_end.startswith(line_buffer) and in_code_block:
                            in_code_block = False
                            line_buffer = ""  # 清空行缓冲区
                            continue

                        # 如果不在代码块中，则打印行缓冲区内容
                        if (not in_code_block and line_buffer) or line_buffer.startswith('#'):
                            # yield char.replace('#', '#><;').replace("'", '')
                            yield_list.append(char.replace('#', '#><;').replace("'", ''))

                            # time.sleep(0.1)  # 模拟逐字打印的效果

                        # 如果遇到换行符，重置line_buffer
                        if '\n' in char:
                            line_buffer = ""
            if currentMode!='reasoning':
                total_buffer=total_buffer.replace("id_list_of_entity(",'id_list_of_entity_fast(')
            # print(total_char_list)
            total_buffer = total_buffer.replace('print(','print_function(')
            chat_result = total_buffer
            full_result = chat_result
            processed_response.append({'role': 'assistant', 'content': chat_result})
            messages.append({'role': 'assistant', 'content': chat_result})
            # print("complete: ", compelete)
            print(full_result)

            if "```python" in full_result and ".env" not in full_result and "pip install" not in full_result:
                steps += 1
                code_list.extend(extract_code_blocks(full_result))

            # print(code_list)
        for line_num, lines in enumerate(code_list):

            # yield "\n\n`Code running...`\n"
            yield_list.append("\n\n`Code running...`\n")
            plt_show = False
            if "plt.show()" in lines:
                plt_show = True
                # print("plt_show")
                filename = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                lines = lines.replace("import matplotlib.pyplot as plt",
                                      "import matplotlib.pyplot as plt\nfrom matplotlib.font_manager import FontProperties\nfont = FontProperties(fname=r'static\msyh.ttc')\n")
                lines = lines.replace("plt.show()", f"plt.tight_layout()\nplt.savefig('static/{filename}')")
                lines = lines.replace("plt.figure(figsize=(10, 6))", f"plt.figure(figsize=(5, 5))")

                # print(lines)

            code_str = process_text_2code(lines,session,sid)

            # print(code_str)
            sys.stdout = output
            start_time = time.time()  # 记录函数开始时间

            try:

                exec(code_str, globals())

            except Exception as e:
                exc_info = traceback.format_exc()
                # 打印错误信息和代码行

                if session['template'] == True:
                    print(e)
                    print(f"An error occurred: \n{exc_info}")
                else:
                    # print(f"An error occurred: {repr(e)}\n{exc_info}")
                    print(f"An error occurred: \n{exc_info}")

                    # print("Nothing can I get! Please change an area and search again :)")
                # print(f"An error occurred: {repr(e)}\n{exc_info}")
            session.modified = True

            end_time = time.time()  # 记录函数结束时间
            run_time = end_time - start_time
            code_result = str(output.getvalue().replace('\00', ''))
            output.truncate(0)
            sys.stdout = original_stdout
            code_result = str(code_result)
            if plt_show and "An error occurred: " not in code_result:
                if not os.path.exists(file_path):
                    filename = 'plot_20240711162140.png'

                code_result = f'![matplotlib_diagram](/static/{filename} "matplotlib_diagram")'
                whole_step = 5  # 确保图返回结果只会被描述一次

                yield_list.append(code_result)
            show_template = details_span(code_result, run_time)

            yield_list.append(list(show_template.values())[0])

            if 'error' in show_template or 'Nothing can I get! Please change an area and search again' in show_template:
                return
            send_result = "code_result:" + short_response(code_result)
            # print(send_result)
            messages.append({"role": "user",
                             "content": send_result})
            processed_response.append({'role': 'user', 'content': send_result})

        send_data(processed_response, sid=sid)

        data_with_response = {
            'len': str(len(messages)),
            'time': str(datetime.now()),
            'ip': str(session['ip_']),
            'user': str(data),
            'os': str(session['os']),
            'browser': str(session['browser']),
            'device': str(session['device_type']),
            'sid': str(sid),
            'yield_list': str(yield_list),
            'answer': processed_response
        }

        formatted_data = json.dumps(data_with_response, indent=2, ensure_ascii=False)

        # 写入文本文件
        with open('static/data3.txt', 'a', encoding='utf-8') as file:
            file.write(formatted_data)
        return yield_list

    yield_list = process_code(data)

    def generate():
        if yield_list != None and yield_list != []:
            for yield_element in yield_list:
                yield yield_element

    return Response(stream_with_context(generate()))

def read_colum_name(text):
    # print(text)
    ask_prompt = """
I will give you a piece of geojson text, and you need to return a json, tell me the key names that can represent the elements category, name, geom:
Do not use the 'id' key as any of the key values
Do not give key like 'key.label', do not give higher level key, just give the direct label

If you are not sure, give the most likely answer

give me a short logic reasoning before giving the result
please give the json format like below:
```json
{
"category": "Example Category",
"name": "name",
"geom": "geometry",
}
```
    """
    result_json = general_gpt_without_memory(query=text, ask_prompt=ask_prompt, json_mode='json_few_shot',verbose=True)
    return result_json


def find_key_values(json_obj, target_key):
    result = []

    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key == target_key:
                if not isinstance(value, str):
                    result.append(value)
                else:
                    result.append(value.replace("_"," "))

            elif isinstance(value, (dict, list)):
                result.extend(find_key_values(value, target_key))
    elif isinstance(json_obj, list):
        for item in json_obj:
            result.extend(find_key_values(item, target_key))

    return result


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    return jsonify({'file_path': file_path})


def convert_to_wkt_4326(geojson_list,epsg='3857'):
    # 创建坐标转换器（假设输入坐标是 EPSG:3857，Web Mercator）
    transformer = Transformer.from_crs(f"EPSG:{str(epsg)}", "EPSG:4326", always_xy=True)

    wkt_list = []

    for geojson in geojson_list:
        geom_type = geojson.get("type")
        coordinates = geojson.get("coordinates")

        if not coordinates:
            raise ValueError("Invalid GeoJSON: Missing coordinates.")

        def transform_coords(coords):
            if isinstance(coords[0], (int, float)):
                x, y = coords[:2]
                return transformer.transform(x, y)
            return [transform_coords(c) for c in coords]

        transformed_coords = transform_coords(coordinates)
        shapely_geom = shape({"type": geom_type, "coordinates": transformed_coords})
        wkt_list.append(shapely_geom.wkt)

    return wkt_list


@app.route('/read_file', methods=['POST'])
def read_file():
    data = request.json
    file_path = data.get('file_path')
    session['file_path']=file_path
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    with open(file_path, 'r',encoding="utf-8") as f:
        content = f.read()
        result_json=read_colum_name(content[:1200])

    if result_json:
        result_json['epsg']=get_geojson_epsg(file_path)

        # Simulate processing the file and returning a dictionary

        return jsonify(result_json)



@app.route('/read_result', methods=['POST'])
def read_result():
    data = request.json
    with open( data['file_path'], 'r',encoding="utf-8") as f:
        content = f.read()
    table_name = data['table'].lower()
    if table_name  in session['col_name_mapping_dict'] or table_name in all_fclass_set or table_name in all_name_set:
        raise Exception ("table name has been used!")
    del data['table']
    content_json = json.loads(content)
    # Simulate processing the form data
    store_json = {}
    for each_col, each_key in data.items():
        if each_col not in ['epsg','table','file_path']:
            store_json[each_col] = find_key_values(content_json, each_key)
            print(len(find_key_values(content_json, each_key)), each_col)

    store_json['geom'] = convert_to_wkt_4326(store_json['geom'],data['epsg'])
    store_json['fclass'] = store_json.pop('category')
    store_json['osm_id'] =range(len(store_json['fclass'] ))

    create_table_from_json(store_json,table_name)
    session['col_name_mapping_dict'][table_name]={}
    col_name_mapping_dict[table_name]={
        "osm_id": "osm_id",
        "fclass": "fclass",
        "name": "name",
        "select_query": f"SELECT uploaded_{table_name} AS source_table, fclass,name,osm_id,geom",
        "graph_name": f"uploaded_{table_name}"
    }
    each_set = ids_of_attribute(table_name)
    fclass_dict_4_similarity[table_name] = each_set
    all_fclass_set.update(each_set)

    each_set = ids_of_attribute(table_name, 'name')
    name_dict_4_similarity[table_name] = each_set
    all_name_set.update(each_set)
    print('Received form data:', data)
    return jsonify({'wait': True})

def get_geojson_epsg(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # GeoJSON 可能包含 "crs" 字段，旧版格式使用 "properties.name"
    crs = data.get('crs', {}).get('properties', {}).get('name')

    if crs:
        try:
            epsg = pyproj.CRS(crs).to_epsg()
            return epsg if epsg else f"无法解析 EPSG 代码: {crs}"
        except Exception as e:
            return f"解析 CRS 失败: {e}"
    else:
        return "GeoJSON 未提供 CRS 信息"
def query_ip_location(ip):
    try:
        ip = ip.replace(" ", "")
        url = f"http://ip-api.com/json/{ip}"
        response = requests.get(url)
        data = json.loads(response.text)

        if data["status"] == "success":
            country = data["country"]
            city = data["city"]
            region = data["regionName"]
            isp = data["isp"]
            res = str(region + "  " + city + "  " + isp)
            return res
        else:
            return ip
    except:
        return ip
# chroma run --path ./chroma_db --host 0.0.0.0

if __name__ == '__main__':
    socketio.run(app,  allow_unsafe_werkzeug=True, host='0.0.0.0', port=9090)
