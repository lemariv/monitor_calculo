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
groq_api_key = os.getenv("GROQ_API_KEY")

# conexao com o provedor de IA (Groq - rapido e gratuito)
conexao = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
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

    system_prompt = """Você é um monitor universitário especialista e exclusivo de Cálculo 1.
    
    1. CUMPRIMENTOS E SAUDAÇÕES:
        - Se o usuário apenas cumprimentar (ex: "Oi", "Olá", "Tudo bem?"), responda de forma breve, educada e receptiva, perguntando qual é a dúvida de Cálculo 1.
        - NUNCA enumere passos nem pergunte se entendeu passos quando for apenas uma saudação.
        - Exemplo:
          Usuário: Olá, tudo bem?
          Assistente: Olá! Tudo ótimo por aqui. Qual é a sua dúvida de Cálculo 1 hoje?

    2. DÚVIDAS E QUESTÕES DE CÁLCULO 1:
        - Explique de forma paciente, clara e didática.
        - Ao resolver exercícios, estruture a resposta de maneira organizada com passos claros (Passo 1, Passo 2, etc.).
        - Somente ao final de uma resolução ou explicação de cálculo, pergunte se o aluno compreendeu todos os passos ou se deseja que algum ponto seja mais detalhado.
        - Se o aluno digitar algo com notação simplificada (ex: 'lim x->0 sen(x)/x'), interprete matematicamente com clareza.

    3. DESPEDIDAS E AGRADECIMENTOS:
        - Se o usuário agradecer ou se despedir, responda cordialmente e se coloque à disposição para futuras dúvidas de Cálculo.
        - Exemplo:
          Usuário: Obrigado pela ajuda!
          Assistente: De nada! Fico feliz em ajudar. Bons estudos e, se surgir mais alguma dúvida de Cálculo 1, é só me chamar!

    4. OUTROS ASSUNTOS E ESCOPO:
        - Seu escopo é ESTRITAMENTE Cálculo 1.
        - Se o usuário perguntar sobre outras disciplinas ou assuntos gerais, informe educadamente que você é um monitor exclusivo de Cálculo 1 e não pode ajudar com outros temas. NUNCA diga que pode conversar sobre qualquer assunto.

    5. FORMATAÇÃO MATEMÁTICA E ESPAÇAMENTO VISUAL (LATEX):
        - Use cifrões para TODAS as fórmulas matemáticas e variáveis.
        - Fórmulas na mesma linha: use um único cifrão (ex: $f'(x) = 2x$).
        - Fórmulas em destaque / bloco: use dois cifrões (ex: $$ \lim_{x \to 0} \frac{\sin(x)}{x} = 1 $$).
        - NUNCA use \\[ \\] nem \\( \\) para matemática. Use ESTRITAMENTE o símbolo de cifrão ($ e $$).
        - ESPAÇAMENTO VISUAL AREJADO (MUITO IMPORTANTE):
          * Sempre pule uma linha em branco ANTES e DEPOIS de cada bloco matemático ($$).
          * Use títulos destacados para cada etapa (ex: ### Passo 1: Identificar a regra, ### Passo 2: Derivar os termos).
          * Use linhas divisórias (---) entre as etapas principais da resolução para o texto não ficar amontoado.
          * NUNCA coloque várias linhas de equações grudadas uma na outra sem uma linha em branco entre elas.
    """

    # lista de memoria, guarda as msgm antigas
    if "historico" not in st.session_state:
        st.session_state.historico = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Olá! Qual é a sua dúvida de Cálculo 1?"}
        ]
    else:
        # Garante que o system prompt atualizado seja sempre usado
        st.session_state.historico[0] = {"role": "system", "content": system_prompt}
    

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
        simil = consulta.similarity_search(prompt, k=5) 

        textos_apoio = "" 
        for i, doc in enumerate(simil): 
            textos_apoio += doc.page_content + "\n\n"

        resposta_modelo = f"""Use os documentos abaixo como material de apoio para responder os alunos sobre Cálculo 1. Se os documentos cobrirem o assunto, baseie-se neles. Se os documentos NÃO cobrirem o assunto mas ainda for uma dúvida de Cálculo 1, responda normalmente com seu próprio conhecimento, sem mencionar que os documentos não contêm o assunto. NUNCA diga ao aluno que os documentos fornecidos não abordam o tema. Se a mensagem for apenas um cumprimento ou despedida, responda sem usar os documentos. Se a mensagem for referente a outra área de ensino, diga que não é especialista nisso.

       
        Documentos: {textos_apoio}
        Pergunta do aluno: {prompt}
        """

        mensagens_conversa = st.session_state.historico[1:] #pega as mensagens tirando a posicao 0, que é o system prompt
        ultimas_mensagens = mensagens_conversa[-6:]  # pega as últimas 6 mensagens do histórico, ignorando a primeira mensagem do sistema
        mensagens_para_api = [st.session_state.historico[0]] + ultimas_mensagens #junta as 6 ultimas mensagens com o prompt

        mensagens_para_api[-1] = {"role": "user", "content": resposta_modelo}  # substitui a última mensagem do usuário pelo conteúdo atualizado com os documentos de apoio
        with st.chat_message("assistant"):
            try:
                resposta_andamento = conexao.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=mensagens_para_api,
                    max_tokens=900
                )
                resposta_processada = resposta_andamento.choices[0].message.content #pega a resposta do modelo, poe na posicao 0 (a primeira), tira os espacos em branco
                st.markdown(resposta_processada) #exibe a resposta do modelo no chat
                st.session_state.historico.append({"role": "assistant", "content": resposta_processada}) #adc a resposta do modelo no historico

            except Exception as e:
                st.error("Falha na comunicação com o servidor de IA")
                st.error(str(e))

                if hasattr(e, 'response'):
                    try:
                        st.error(e.response.json())
                    except:
                        st.error(e.response.text)


conversa()