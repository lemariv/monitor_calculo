import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import io
from huggingface_hub import HfApi
import datetime



load_dotenv()
hf_token = os.getenv("HF_TOKEN")
hf_dataset_token = os.getenv("HF_DATASET_TOKEN")

# conexao com o Hugging Face
conexao = OpenAI(
    base_url="https://router.huggingface.co/v1/",
    api_key=hf_token
)

@st.cache_resource
def carregar_gaveta():
    tradutor_emb1 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #o banco faiss salvo pega as pastas que foram criadas no banco de dados
    #tradutor_emb1 - usa o msm tradutor de converter os documentos em vetores
    #allow_dangerous_deserialization=True confirma que o arquivo .pkl foi criado de forma autorizada
    return FAISS.load_local("banco_faiss_salvo", tradutor_emb1, allow_dangerous_deserialization=True)

consulta = carregar_gaveta()

def conversa():

    st.title("Monitor Virtual")

    system_prompt = """Você é um monitor universitário especialista em cálculo 1. 
    
            REGRAS ABSOLUTAS DE INÍCIO DE RESPOSTA:
                - SEMPRE que o usuário iniciar a conversa com um cumprimento, responda com outro cumprimento adequeado.
                    Exemplo: Usuário: Olá, tudo bem?
                    Assistente: Olá! Tudo ótimo, e você? Qual sua dúvida de cálculo hoje?
                - Se o usuário iniciar a conversa com uma dúvida geral,responda com uma saudação e explique o assunto da dúvida de forma mais geral, e por fim, pergunte se ele quer que especifique alguma parte da explicação.
                    Exemplo: Usuário: O que é derivada?
                    Assistente: Olá! A derivada é um conceito fundamental em cálculo que mede a taxa de variação de uma função em relação a uma variável. Se você quiser, posso detalhar mais sobre como calcular derivadas e suas aplicações.
                - Se o usuário iniciar a conversa com uma dúvida específica, responda com uma saudação e explique detalhadamente o passo a passo da resolução da questão, e por fim, pergunte se ele quer que especifique algum passo da explicação.
                    Exemplo: Usuário: Como calcular a derivada de x^2?
                    Assistente: Olá! Para calcular essa derivada, aplicamos a regra da potência. O passo a passo é:
                    $$
                    \frac{d}{dx}(x^2) = 2x
                    $$. Portanto, a derivada de x^2 é 2x. Entendeu todos os passos ou deseja que eu explique algum deles com mais detalhes?
    
    
            REGRAS ABSOLUTAS DE FINAL DE RESPOSTA:
                - É ABSOLUTAMENTE OBRIGATÓRIO que você pergunte ao usuário se ele entendeu a explicação ou se deseja que você explique algum passo com mais detalhes ao final.
                - SEMPRE que o usuário finalizar a conversa com um agradecimento, responda com outro agradecimento e se coloque à disposição para futuras dúvidas.
                    Exemplo: Usuário: Obrigado pela ajuda!
                    Assistente: De nada! Fico feliz em poder ajudar. Se tiver mais dúvidas no futuro, pode me procurar. Até mais!
            
            REGRAS ABSOLUTAS DE RESOLUÇÕES DE QUESTÕES:
                - EM TODO CASO de resolução de questões, ENUMERE os passos da resposta.
                - Comece o Passo 1 explicando de forma clara o que a questão quer. 
                Exemplo: Passo 1 - Nessa questão, pede-se para achar o limite da função quando x tende a 0. Passo 2 - Na substiuição direta na função vemos que o limite é indeterminado. Passo 3 - Fazendo as substituições algébricas ....
    
    
            REGRAS ABSOLUTAS DE COMPORTAMENTO:
                - Você DEVE resolver o passo a passo das dúvidas do usuário antes de formular e mandar a resposta final.
                - SEMPRE explique detalhadamente o passo a passo da resolução, mesmo que o usuário não peça.
                - Seja paciente, explicativo e diga o passo a passo das resoluções para os alunos.
                - Quando a pergunta ou dúvida for de cálculo, explique detalhadamente o passo a passo da resolução;
                - Analise a finalidade da pergunta do usuário e responda conforme o que for pedido.
                - SEMPRE gere explicações a cada passo de exemplos de questões, e, se for pedido pelo usuário, gere outra resposta posterior mais detalhada sobre cada passo.
                - Se o usuário, na sua pergunta, digitar algo confuso como 'lim x->0 sen/x2', você deve interpretar a pergunta como 'limite de sen(x)/x^2 quando x tende a 0' e perguntar se ele quis dizer isso. Logo após, responder com o que foi respondido pelo usuário.
    
    
    
             REGRAS ABSOLUTAS DE FORMATAÇÃO:
                - Você DEVE usar cifrões para TODAS as fórmulas matemáticas e numéricas.
                - Para fórmulas no meio do texto, use um único cifrão. Exemplo: $x = 2$.
                - Para fórmulas em blocos separados, use dois cifrões. Exemplo: $$ lim x \to 1 $$
                - É ESTRITAMENTE PROIBIDO encapsular comandos matemáticos dentro de parênteses comuns.
                - Nunca envie código solto sem os cifrões ao redor.
                - NUNCA use \[ ou \] nem \( ou \) para matemática. Use ESTRITAMENTE o símbolo de cifrão: $ para equações na mesma linha e $$ para equações em bloco.
                    
            REGRAS ABSOLUTAS DE CONTRA-COMPORTAMENTO:
                - É ESTRITAMENTE PROIBIDO que você diga que não sabe a resposta ou que não pode responder a pergunta do usuário.
                - É ESTRITAMENTE PROIBIDO que você diga que não entendeu a pergunta do usuário.
                - É ESTRITAMENTE PROIBIDO que você diga que não tem informações suficientes para responder a pergunta do usuário.
                - Caso o usuário faça uma pergunta de física, química, biologia ou qualquer outra disciplina que não seja cálculo, você DEVE responder que não é especialista na área e que só pode responder dúvidas de cálculo 1.


                    EXEMPLO DE COMPORTAMENTO ESPERADO:
                            Usuário: Oi, Calcule o limite de x^2 quando x tende a 4.
                            Assistente: Olá, tudo bem? Para calcular esse limite, substituímos o valor diretamente na função. O passo a passo é:
                            $$
                            \lim_ x \to 4 x^2 = 4^2 = 16
                            $$. Portanto, o resultado final é $16$. Entendeu todos os passos ou deseja que eu explique algum deles com mais detalhes?
    
    
                    """

    # lista de memoria, guarda as msgm antigas
    if "historico" not in st.session_state:
            #sao dicionarios, as keys 
            st.session_state.historico = [{"role":"system", "content": system_prompt
                },
                {"role": "assistant", "content": "Olá! Qual é a sua dúvida?"}]
    

    with st.sidebar: #comando que joga o que está abaixo dele em uma barra esquerda
        st.header("Menu")
        st.divider() #linha de separação

        if st.button("Nova Conversa", use_container_width=True): #botão que limpa o historico, e cria uma nova conversa

            if len(st.session_state.historico) > 2: #se o historico de conversa do aluno foi +2 mensagens, entao suas conversas serão salvas
                try:
                    df = pd.DataFrame(st.session_state.historico[1:]) #o método dataframe transforma o historico em uma tabela, ignora o 1 item do historico (prompt)

                    csv_buffer = io.BytesIO() #io contra entradas e saidas de dados. 
                    df.to_csv(csv_buffer, index=False) #a biblioteca pandas escreve o historico de conversas no csv_buffer
                    csv_buffer.seek(0) #leitura de cima pra baixo

                    api = HfApi(token=hf_dataset_token)

                    tempo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_arquivo = f"conversa_{tempo}.csv"

                    api.upload_file(
                        path_or_fileobj=csv_buffer,
                        path_in_repo=f"conversas/{nome_arquivo}",
                        repo_id="leticia2006/historico-bot",
                        repo_type="dataset")
                except Exception as e:
                    print(f"Erro ao salvar histórico: {e}")

            st.session_state.historico = [
                {"role":"system", "content": system_prompt},
                {"role": "assistant", "content": "Olá! Qual é a sua dúvida?"}
            ]
            st.rerun()

        st.subheader("Dúvidas recentes")

        duvidas_rec = [msg for msg in st.session_state.historico if msg["role"] == "user"] #pega as msgms do user, que sao as duvidas recentes

        if len(duvidas_rec) == 0: #para quando o user nao tiver digitado nada ainda
            st.write("Nenhuma dúvida recente.")
        else:
            for i, msg in enumerate(duvidas_rec): #pega apenas o que o user digitou (50 caracteres)
                resumo_duv = msg["content"][:50].replace("\n", " ")
                st.markdown(f"**{i+1}.** {resumo_duv}...") #exibe as duvidas recentes, com o numero e o resumo da msgm, que é os 50 primeiros caracteres da msgm, e substitui os \n por espaço

                st.divider()

        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True) 
        
        # Cria as letrinhas miúdas estáticas
        st.markdown("""
            <div style="text-align: center; color: gray; font-size: 11px;">
                © 2026 Monitor Virtual.<br>
                Conceitos adaptados de STEWART, J. Cálculo, Vol 1.<br>
                Materiais cedidos com autorização pelo Professor e Monitor da disciplina.
            </div>
        """, unsafe_allow_html=True)


    
    #baloes antigos do historico
    for mensagem in st.session_state.historico: #p cada msgm antiga, o for faz um balao
        if mensagem["role"] != "system": #se for msgm do sistema, nao exibe
            with st.chat_message(mensagem["role"]): #titulo do balao, se é user ou assistente
                st.write(mensagem["content"]) #corpo do balao, o texto da msgm

    #se o user digitou, exibe a mensagem
    if prompt := st.chat_input("Digite sua dúvida de cálculo..."): #guarda o prompt no st.chat caso tuver alguma mensagem dentro dela, digitada pelo user 
        with st.chat_message("user"):
            st.markdown(prompt) #medoto que deixa o texto do balao de msgm do user, o markdown é formatado para fórmula
            st.session_state.historico.append({"role": "user", "content": prompt})
            #acessa, localiza o historico, adc a msgm do user, com o papel e o texto digitado

        #pesquisa por similaridade 
        simil = consulta.similarity_search(prompt, k=3) 

        textos_apoio = "" 
        for i, doc in enumerate(simil): 
            textos_apoio += doc.page_content + "\n\n"

        resposta_modelo = f"""Use os documentos abaixo para responder os alunos, caso sejam perguntas referentes à Cálculo 1. Se a mensagem for apenas um cumprimento ou despedida, responder sem usar os documentos abaixo. Se a mensagem for referente a outra área de ensino, não use os documentos abaixo e diga que não é especialista nisso.

       
        Documentos: {textos_apoio}
        Pergunta do aluno: {prompt}
        """

        mensagens_conversa = st.session_state.historico[1:] #pega as mensagens tirando a posicao 0, que é o system prompt
        ultimas_mensagens = mensagens_conversa[-6:]  # pega as últimas 6 mensagens do histórico, ignorando a primeira mensagem do sistema
        mensagens_para_api = [st.session_state.historico[0]] + ultimas_mensagens #junta as 6 ultimas mensagens com o prompt

        mensagens_para_api[-1] = {"role": "user", "content": resposta_modelo}  # substitui a última mensagem do usuário pelo conteúdo atualizado com os documentos de apoio
        
        with st.chat_message("assistant"):
            #resposta do modelo, usando a api da openai, que leva o historico de msgms, o modelo e o maximo de tokens que o modelo pode mostrar no chat
            try:
                resposta_andamento = conexao.chat.completions.create( #o sistema da openai leva a mensagem do user e retorna uma resposta do modelo, leva o historico, o modelo e o maximo de tokens que o modelo pode mostrar no chat
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    messages=mensagens_para_api #ia recebe tudo, o historico, o system, o assistant e user
                )
                resposta_processada = resposta_andamento.choices[0].message.content #pega a resposta do modelo, poe na posicao 0 (a primeira), tira os espacos em branco
                st.markdown(resposta_processada) #exibe a resposta do modelo no chat
                st.session_state.historico.append({"role": "assistant", "content": resposta_processada}) #adc a resposta do modelo no historico

            except Exception as e:
                st.error("Falha na comunicação com o Hugging Face")
                st.error(str(e))

                if hasattr(e, 'response'):
                    try:
                        st.error(e.response.json())
                    except:
                        st.error(e.response.text)


conversa()