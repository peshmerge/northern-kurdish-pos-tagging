from utils import extract_tags
from utils import extract_tokens
from markupsafe import Markup
from flask import Flask, request
from pos_model import POSModel
import bleach

app = Flask(__name__)


@app.route('/')
def home():
    return """
    <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <meta name="color-scheme" content="light dark" />
            <title>Northern Kurdish POS tagging by Peshmerge Morad</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css"/>
            <style>
                .ADJ {
                   background-color: #87CEEB
                }
                .PRON {
                   background-color: #FFFF00
                }
                .NOUN {
                   background-color: #D3D3D3
                }
                .VERB {
                   background-color: #6B8E23
                }
                .DET {
                   background-color: #6495ED
                }
                .PART {
                   background-color: #FFD700
                }
                .IZAFE {
                   background-color: #7B68EE
                }
                .PUNCT {
                   background-color: #BDB76B
                }
                .ADV {
                   background-color: #BC8F8F
                }
                .CCONJ {
                   background-color: #8FBC8F
                }
                .PROPN {
                   background-color: #F08080
                }
                .NUM {
                   background-color: #F4A460
                }
                .ADP {
                   background-color: #DDA0DD
                }
                .AUX {
                   background-color: #40E0D0
                }
                .SCONJ {
                   background-color: #98FB98
                }
                
                table {
                    border-collapse: collapse; /* Collapse borders for table cells */
                    width: 100%; /* Set table width to fill the container */
                }
        
                th, td {
                    border: 5px solid white; /* Add border for visual distinction */
                    padding: 10px; /* Add padding for spacing within table cells */
                    text-align: center; /* Center text horizontally */
                }
        
                .word {
                    background-color: lightblue; /* Background color for word row */
                }
        
                .tag-row {
                    background-color: lightgreen; /* Background color for tag row */
                }
                .arrow-cell{
                    font-size:1.5rem !important;
                }
            </style>
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
            <script>
                $(document).ready(function(){
                    $('form').submit(function(event){
                        event.preventDefault();
                        $('input[type=submit]').prop('disabled', true);
                        $('input[type=submit]').prop('value',"Running the model.....");
                        $.ajax({
                            type: 'POST',
                            url: '/pos_tag',
                            data: $('form').serialize(),
                            success: function(response){
                            //alert(response);
                            $('#result').html(response);
                            },
                            complete: function() {
                            $('input[type=submit]').prop('disabled',false);
                            $('input[type=submit]').prop('value',"Submit");
                            
                            },
                            error: function(xhr, status, error) {
                                alert('Error: ' + error);
                            }
                        });
                    });
                });
            </script>
    </head>
    <body style="padding: 2rem !important;">
        <h1>Northern Kurdish POS tagging</h1>
         <form>
        <div class="grid">
            <div> 
                <label>Training Data type:</label>
                <input type="radio" name="training_data_type" value="augmented" checked>Augmented<br>
                <input type="radio" name="training_data_type" value="original">Original
            </div>

            <div> 
                <label>Tokenization Method:</label>
                <input type="radio" name="tokenization_method" value="KLPT" checked>KLPT<br>
                <input type="radio" name="tokenization_method" value="NLTK">NLTK<br>
                <input type="radio" name="tokenization_method" value="manual">Manual
            </div>
            
             <div> 
                <label>Models:</label>
                <input type="radio" name="model" value="All">All<br>
                <input type="radio" name="model" value="Baseline" checked>Baseline<br>
                <input type="radio" name="model" value="HMM">HMM<br>
                <input type="radio" name="model" value="ExtraTrees">ExtraTrees<br>
                <input type="radio" name="model" value="AveragedPerceptron">AveragedPerceptron<br>
                <input type="radio" name="model" value="BiLSTM">BiLSTM<br>
                <input type="radio" name="model" value="CRF">CRF<br>
                <input type="radio" name="model" value="NK-XLMR">NK-XLMR
            </div>
            <div>
            <label>Output Style:</label>
            <input type="radio" name="output_style" value="graphical" checked>Graphical <br>
            <input type="radio" name="output_style" value="json">JSON 
            </div>
        </div>
            <label>Sentence:</label>
            <input type="text" value="Leyla Qasim dixwest dengê kurdan li cîhanê bide bihîstin." 
            name="sentence" required minlength="3" >
            <input type="submit" value="Submit">
        </form>
        <div id="result">       
        </div>
    </body>
    </html>
    """


def generate_html_for_pos(pos_tokens_tags, model):
    content = f'<h4>{model}</h4>'
    content += '<table>'
    content += '<tr class="word-row">'
    tokens = extract_tokens(pos_tokens_tags)

    tags = extract_tags(pos_tokens_tags)
    for token in tokens:
        content += f'<th class="word">{token}</th>'
    content += '</tr>'
    content += '<tr class="tag-row">'
    for tag in tags:
        content += f'<td class="arrow-cell">&#8593;</td>'
    content += ' </tr>'
    content += '<tr class="tag-row">'
    for tag in tags:
        content += f'<td class="{tag}">{tag}</td>'
    content += ' </tr>  </table>'
    return content


def perform_pos(model, training_data_type, sentence, tokenization_method):
    pos_model = POSModel(model, training_data_type)
    pos_model.load_pos_model()
    return pos_model.predict_pos_tags(sentence, tokenization_method)


@app.route('/pos_tag', methods=['POST'])
def pos_tag():
    training_data_type = bleach.clean(request.form['training_data_type'])
    tokenization_method = bleach.clean(request.form['tokenization_method'])
    model = bleach.clean(request.form['model'])
    sentence = bleach.clean(request.form['sentence'])
    output_style = bleach.clean(request.form['output_style'])
    return_value = ''
    if model == 'All':
        for model in POSModel.POS_MODELS:
            response = perform_pos(model, training_data_type, sentence, tokenization_method)
            if output_style == 'graphical':
                return_value += generate_html_for_pos(response, model)
            else:
                return_value += f'<h3>{model}</h3>'
                return_value += f"<pre>{dict(response)} </pre>"
    else:
        response = perform_pos(model, training_data_type, sentence, tokenization_method)
        if output_style == 'graphical':
            return_value = generate_html_for_pos(response, model)
        else:
            return_value = f"<pre>{dict(response)}</pre>"
    return return_value


if __name__ == '__main__':
    app.run(debug=True)
