
class Color_():
    def color_history(history):
        format_history=[]
        for i in range(0,len(history),2):
            format_history.append({'role': 'user', 'content': history[i]})
            format_history.append({'role': 'assistant', 'metadata': '', 'content': history[i+1]})    
        return format_history




    def color_question(question,history):
        #使用提示词将question和history组合起来
        prompt='''Answer questions based on doc, doc{}, question:{}'''.format(question,history)
        return prompt;