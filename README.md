# Pose Studio

Mude a pose de uma pessoa numa foto **sem mudar a câmera**. Depois exporte as referências para gerar a imagem nova no Nano Banana, no Seedream, no Flux, no ControlNet etc.

Tudo roda no navegador. A foto não é enviada para nenhum servidor.

## Como funciona

1. **Carregue a foto.** O MediaPipe detecta a pose, encaixa um boneco 3D em cima da pessoa e recorta a silhueta dela. Ao mesmo tempo, o Depth Anything V2 calcula o depth map da cena inteira, incluindo mesa, cadeira e objetos.
2. **Alinhar ao original.** Corrija os pontos que a detecção errou. Nesse modo cada ponto se move sozinho.
3. **Nova pose.** Arraste mãos, pés e articulações:
   - os ossos mantêm o comprimento;
   - pulsos e tornozelos usam IK (o cotovelo ou joelho acompanha);
   - a roda do mouse, ou Q/E, aproxima e afasta o ponto da câmera;
   - na vista lateral dá pra ajustar a profundidade arrastando;
   - o esqueleto rosa mostra a pose original.
4. **Exportar.**

| Arquivo | O que é | Onde usar |
|---|---|---|
| `depth` | Depth da cena com a pose nova. As partes que não mudaram mantêm o depth original. | ControlNet Depth, Flux Depth, referência |
| `mask` | Área que muda: membro antigo + membro novo, com margem | Inpainting |
| `mannequin` | Boneco cinza na pose nova, mesmo enquadramento | Imagem de referência de pose (Nano Banana, Seedream) |
| `guide` | Foto original com o membro antigo apagado e o boneco na posição nova | Imagem de referência para edição |
| `openpose` | Esqueleto no padrão OpenPose (COCO-18) | ControlNet OpenPose |
| `compare` | Folha BEFORE / AFTER / CHANGE com foto, depth, boneco e/ou OpenPose. Vermelho = sai, verde = entra, setas = movimento de cada articulação. | Imagem de referência que explica a mudança (Nano Banana, Seedream) |
| `pose.json` | Coordenadas 3D das articulações (original e nova) | Reabrir ou depurar |

O painel também sugere um prompt de edição em inglês, montado a partir das partes que você mexeu.

## Rodar localmente

Não tem etapa de build, mas os módulos ES precisam de um servidor HTTP:

```bash
python3 -m http.server 8000
# abra http://localhost:8000
```

## Publicar no GitHub Pages

Settings → Pages → *Deploy from a branch* → escolha a branch e a pasta `/ (root)`.

## Limitações conhecidas

- Objetos na mão (cartão, celular) não acompanham o movimento.
- Roupa solta é tratada como corpo. O boneco não modela saia ou manga larga.
- Membros escondidos (pernas cruzadas, braço atrás do corpo) costumam precisar de ajuste manual.
- O depth do Depth Anything é relativo. O boneco é calibrado pelo depth original da própria pessoa, e o controle "Profundidade do boneco" faz o ajuste fino.

## Créditos

- [three.js](https://threejs.org) (MIT)
- [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) (Apache 2.0)
- [Transformers.js](https://github.com/huggingface/transformers.js) (Apache 2.0)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2): Small é Apache 2.0; Base é CC BY-NC 4.0, ou seja, **uso não comercial**.
