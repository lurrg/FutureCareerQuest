from flask import Flask, render_template, request
import utils as ut


app = Flask(__name__)

# 连接到 MongoDB 数据库
db = ut.connect_mongodb()

# Set up embedding model
emb_model = ut.set_emb_model()

# 根路由，渲染主页
@app.route('/')
def home():
    # 初次加载时，job_title 和 skills 为空
    return render_template('mainpage.html', job_title="", skills="", coreNodesCnt=5)

# 处理表单提交的路由
@app.route('/submit', methods=['POST'])
def submit():
    job_title = request.form.get('jobTitle')
    skills = request.form.get('skills')
    coreNodesCnt_str = request.form.get('coreNodesCnt')

    # Ensure coreNodesCnt is an integer and has a default value
    try:
        coreNodesCnt = int(coreNodesCnt_str) if coreNodesCnt_str else 5
    except ValueError:
        coreNodesCnt = 5  # Default to 5 if conversion fails

    # 如果用户没有提交表单，则返回原始网页
    if not job_title and not skills:
        return render_template('mainpage.html', job_title="", skills="", coreNodesCnt=coreNodesCnt, error_message="Please enter a job title or skills.")

    # 获取相似职位
    top_similar_roles = ut.get_similar_roles(job_title, skills, similarity_bar=0.3)

    if top_similar_roles is None:
        return render_template('mainpage.html', job_title=job_title, skills=skills, coreNodesCnt=coreNodesCnt, error_message="No similar positions found.")

    links, nodes, skillsData = ut.get_graph_data(top_similar_roles, top_N=coreNodesCnt)
    table_data = ut.get_table_data(top_similar_roles, top_N=50)

    # 确保数据不是 None
    links = links or []
    nodes = nodes or []
    skillsData = skillsData or {}
    table_data = table_data or {}

    return render_template('mainpage.html', job_title=job_title, skills=skills, coreNodesCnt=coreNodesCnt, links=links, nodes=nodes, skillsData=skillsData, table_data=table_data)

if __name__ == '__main__':
    app.run(debug=True)

