ffmpeg -r 60 -f image2 -i output128/generated-images-%6d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p faces.mp4`
