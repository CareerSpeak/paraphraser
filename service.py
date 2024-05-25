from flask import Flask, request

from paraphraser import Paraphraser

app = Flask(__name__)


@app.route('/', methods=['POST'])
def paraphraser():
    args = request.args

    text = args.get('text')

    paraphrasedText = Paraphraser(text)

    return paraphrasedText


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=65535)
