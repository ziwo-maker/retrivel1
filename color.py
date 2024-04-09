


def color_history_chat(history):
        format_history=[]
        for i in range(0,len(history),2):
            format_history.append({'role': 'user', 'content': history[i]})
            format_history.append({'role': 'assistant', 'metadata': '', 'content': history[i+1]})    
        return format_history
def colo_histoty(history):
    message=[];
    if(len(history)==1):
         return {"role": "user", "content": history[0]}
    for i in range(0,len(history),2):
        message.append({"role": "user", "content": history[i]})
        message.append({"role": "assistant", "content": history[i+1]})
    return message;