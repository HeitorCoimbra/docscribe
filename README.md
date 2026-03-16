# 🏥 DocScribe

Sistema para extração de sumários estruturados de pacientes de UTI a partir de áudios de passagem de plantão.

## Arquitetura

```
Áudio → Groq Whisper (transcrição) → Claude (análise) → Sumário estruturado
```

## Aplicações

### 📱 Streamlit (Protótipo)

Interface simples para testes rápidos.

```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

### 💬 Chainlit (Produção)

Interface conversacional completa com:
- ✅ OAuth (Google/GitHub)
- ✅ Histórico de sessões (PostgreSQL)
- ✅ Sessões agrupadas por dia
- ✅ Títulos automáticos "Leito X - Nome"

```bash
cd chainlit
pip install -r requirements.txt
chainlit run app.py
```

## Estrutura de Pastas

```
docscribe/
├── streamlit/              # App Streamlit (protótipo)
│   ├── app.py              # Interface principal
│   ├── app_chat.py         # Interface de chat
│   ├── core.py             # Lógica compartilhada
│   └── requirements.txt
│
├── chainlit/               # App Chainlit (produção)
│   ├── app.py              # App principal com OAuth
│   ├── core.py             # Lógica compartilhada
│   ├── database.py         # Modelos PostgreSQL
│   ├── chainlit.md         # Página de boas-vindas
│   ├── Dockerfile          # Para Railway
│   ├── railway.json        # Config Railway
│   └── requirements.txt
│
└── README.md
```

## Formato do Sumário

```
Leito [X] - [Nome do Paciente]

Diagnósticos:
1- [diagnóstico principal]
2- [outros diagnósticos]

Pendências:
1- [pendência mais urgente]
2- [outras pendências]

Condutas:
• [Manter...] (verbo no infinitivo)
• [Iniciar...] (verbo no infinitivo)
```

## Tecnologias

- **Groq Whisper** - Transcrição de áudio ultrarrápida
- **Anthropic Claude** - Análise e extração estruturada
- **Chainlit** - Interface conversacional
- **PostgreSQL** - Persistência de sessões
- **Railway** - Deploy e hosting

## Custos Estimados

| Serviço | Custo |
|---------|-------|
| Groq Whisper | Gratuito (rate limits) |
| Claude Sonnet | ~$0.003/sumário |
| Railway | ~$5/mês (Hobby) |
| PostgreSQL | Incluído no Railway |

**Total estimado:** ~$5-10/mês para uso moderado
