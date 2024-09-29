import cv2
import numpy as np
import imutils
from scipy.ndimage.filters import gaussian_filter
from ultralytics import YOLO
import time
import os

def main():
    rtsp_url = 'rtsp://admin:a3e1lm2s2y@192.168.15.103:10554/tcp/av0_0'
    print("Conectando à câmera...")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Não foi possível conectar à câmera. Verifique o URL RTSP e a conexão de rede.")
        return
    print("Conexão estabelecida com sucesso.")

    # Carregando o modelo YOLOv8
    print("Carregando o modelo YOLOv8...")
    model = YOLO('yolov8m.pt')  # Escolha o modelo apropriado (nano, small, medium, large)
    print("Modelo carregado com sucesso.")

    heatmap_accumulator = None
    alpha = 0.6  # Fator de transparência para sobreposição

    # Variáveis para controle de tempo e salvamento
    save_interval = 300  # Intervalo de 5 minutos em segundos
    last_save_time = time.time()
    output_folder = 'mapas_de_calor'

    # Criar a pasta se não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Iniciando o processamento dos frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o frame. Verifique a conexão com a câmera.")
            break
        else:
            # Frame capturado com sucesso
            frame = imutils.resize(frame, width=640)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inferência
            results = model.predict(source=rgb_frame, imgsz=640, conf=0.5, iou=0.5)

            # Inicializar o acumulador na primeira iteração
            if heatmap_accumulator is None:
                heatmap_accumulator = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

            # Processar as detecções
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Classe 'pessoa'
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        heatmap_accumulator[y1:y2, x1:x2] += 1

                        # Desenhar retângulo ao redor da pessoa
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Aplicar filtro gaussiano para suavizar o mapa de calor
            heatmap = gaussian_filter(heatmap_accumulator, sigma=15)

            # Normalizar o mapa de calor
            heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_normalized = heatmap_normalized.astype(np.uint8)

            # Aplicar mapa de cores
            heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

            # Sobrepor o mapa de calor ao frame original
            overlay = cv2.addWeighted(frame, alpha, heatmap_color, 1 - alpha, 0)

            # Verificar se já passou o intervalo de tempo para salvar a imagem
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                # Salvar o snapshot com o mapa de calor aplicado
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                filename = f'snapshot_mapa_calor_{timestamp}.png'
                filepath = os.path.join(output_folder, filename)
                cv2.imwrite(filepath, overlay)
                print(f"Imagem do snapshot com mapa de calor salva em: {filepath}")
                last_save_time = current_time  # Atualizar o tempo do último salvamento

            cv2.imshow('Mapa de Calor', overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Encerrando o processamento.")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
