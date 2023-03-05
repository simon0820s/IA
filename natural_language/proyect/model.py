import openai

openai.api_key="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"

def run(text):
    question="tomando en cuenta el siguiente texto : "
    question+=text
    question+="\n extrae los siguientes elementos: Demandado, Demandante, numero del radicado, fecha del remate, hora del remate, secuestre, departamento, ciudad y el juzgado que va a ejecutar el remate"
    complection=openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=2048)
    
    response=complection.choices[0].text
    
    return response