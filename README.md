# 📚 Mitologia Grega — RAG com LangChain + ChromaDB

Um mini projeto de **Retrieval-Augmented Generation (RAG)** que permite fazer perguntas em linguagem natural sobre um PDF de mitologia grega, combinando embeddings semânticos com um LLM da OpenAI para gerar respostas contextualizadas.

---

## 🧠 Como funciona

1. O PDF `mitologia_grega.pdf` é carregado e dividido em chunks de texto.
2. Cada chunk é transformado em um embedding vetorial usando o modelo `text-embedding-3-large` da OpenAI.
3. Os embeddings são persistidos localmente em um banco de dados vetorial (ChromaDB).
4. Para cada pergunta, os chunks mais relevantes são recuperados via busca por similaridade.
5. O LLM (`gpt-3.5-turbo`) recebe os chunks recuperados como contexto e gera a resposta final.

```
PDF → Chunks → Embeddings → ChromaDB
                                 ↓
Pergunta → Retriever → Contexto → LLM → Resposta
```

---

## 🗂️ Estrutura do projeto

```
.
├── main.py                         # Script principal
├── mitologia_grega.pdf             # PDF de origem (não incluído no repo)
├── vectorstore/
│   └── manual_chroma_small/        # Banco vetorial persistido (gerado automaticamente)
├── .env                            # Variáveis de ambiente (não versionado)
└── README.md
```

---

## ⚙️ Requisitos

- Python 3.9+
- Uma chave de API da [OpenAI](https://platform.openai.com/)

### Dependências

Instale as dependências com:

```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters chromadb pypdf python-dotenv
```

> **Nota:** o pacote `langchain-classic` é necessário para o `RetrievalQA`. Verifique a compatibilidade com a versão do LangChain instalada.

---

## 🚀 Como usar

### 1. Clone o repositório

```bash
git clone https://github.com/Gahbrielms/exercicio_rag.git
cd exercicio_rag
```

### 2. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=''
```

### 3. Adicione o PDF

Coloque o arquivo `mitologia_grega.pdf` na raiz do projeto.

### 4. Execute

```bash
python main.py
```

Na primeira execução, o vectorstore será criado e persistido. Nas execuções seguintes, ele será reutilizado automaticamente.

---

## 💬 Exemplos de perguntas

O script já inclui três perguntas de exemplo:

- *"Me resuma a história de Zeus?"*
- *"Qual a importância do Olimpo no contexto da mitologia grega?"*
- *"Quais são os principais deuses da mitologia grega?"*

Para adicionar suas próprias perguntas, edite a lista `perguntas` em `main.py`.

---

## 🔧 Parâmetros configuráveis

| Parâmetro | Localização | Descrição |
|---|---|---|
| `chunk_size` | `RecursiveCharacterTextSplitter` | Tamanho de cada chunk de texto |
| `chunk_overlap` | `RecursiveCharacterTextSplitter` | Sobreposição entre chunks consecutivos |
| `k` | `as_retriever` | Número de chunks recuperados por consulta |
| `model` | `ChatOpenAI` | Modelo do LLM utilizado |
| `persist_dir` | `main.py` | Diretório de persistência do vectorstore |

---

## 📄 Licença

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
