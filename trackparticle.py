#!/usr/bin/env python3
# track a single wrist, and draw snowflake particles on the screen
import cv2
import time
import mediapipe as mp
import pygame
import particlepy
import sys
import random

# use OpenCV to to get images
cap = cv2.VideoCapture(0)

# configure mediapipe for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

pygame.init()

# pygame config
SIZE = 1024, 768
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Hand Tracking")
# pygame.mouse.set_visible(False)

# timing
clock = pygame.time.Clock()
FPS = 30

# delta time
old_time = time.time()
delta_time = 0

# particle system to manage particles
particle_system = particlepy.particle.ParticleSystem()

images = [
    pygame.image.load("data/blue-snowflake-small.png").convert_alpha(),
    pygame.image.load("data/pToArjn9c-small.png").convert_alpha()
]

# initial 'mouse' position
mouse_pos = [0, 0]

while True:
    # quit window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    # delta time
    now = time.time()
    delta_time = now - old_time
    old_time = now

    # update particle properties
    particle_system.update(delta_time=delta_time)
    # print(len(particle_system.particles))

    success, img = cap.read()
    imgRGB = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)


    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            image_height, image_width, _ = img.shape
            x = handLms.landmark[mpHands.HandLandmark.WRIST].x
            y = handLms.landmark[mpHands.HandLandmark.WRIST].y
            print(
                f'Wrist coordinates: (',
                f'{x}, '
                f'{y})'
                )
            # get mouse position
            mouse_pos = [int(x * 1024), int(y * 768)]
        for _ in range(1):
            particle_system.emit(
                particlepy.particle.Particle(shape=particlepy.shape.Image(surface=random.choice(images), size=(50, 50), alpha=255),
                                            position=mouse_pos,
                                            velocity=(random.uniform(-150, 150), random.uniform(-150, 150)),
                                            delta_radius=0.5))
    # else:
    #     print('Nothing detected')





    # render shapes
    particle_system.make_shape()

    # post shape creation manipulation
    for particle in particle_system.particles:
        particle.shape.angle += 5

    # render particles
    particle_system.render(surface=screen)

    # update display
    pygame.display.update()
    screen.fill((13, 17, 23))
    clock.tick(FPS)


# https://arkalsekar.medium.com/how-to-get-all-the-co-ordinates-of-hand-using-mediapipe-hand-solutions-ac7e2742f702
# https://google.github.io/mediapipe/solutions/hands.html
# https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/