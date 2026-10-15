# Wall photo collection: instructions for a team member

Goal: 200 to 300 photos of real walls, taken the way the rover's camera will see them. The AI model that decides where the rover must not paint (windows, doors, AC units, meter boxes, pipes, grills) has only seen public photos, which are the wrong kind. Your photos are the single biggest thing that will make it work on real walls. You do not need to install anything for Parts 1 to 5. One afternoon is enough.

## Part 1. Get your phone ready (5 minutes)

1. Clean the camera lens with a cloth.
2. Save as **JPG**, not HEIC. iPhone: Settings > Camera > Formats > **Most Compatible**. Android: choose JPG/JPEG (turn off HEIF/RAW).
3. Turn **off** beauty filters, portrait mode, night mode, and any "AI" effects. Use the normal photo mode.
4. Turn **on** the grid lines (Camera settings > Grid). They help you hold the phone square to the wall.
5. Check you have at least 2 GB free storage and a charged phone.

## Part 2. How to take each photo

1. **Stand square to the wall**, 2 to 4 metres away, phone at chest height, held upright or sideways, pointing straight at the middle of the wall. Do not stand at an angle. The wall's edges should look straight, not leaning.
2. **Fill the frame with the wall.** Try to keep all four corners of the wall (or the whole visible wall) in the photo. Very little sky, ground or neighbouring buildings.
3. Take **one sharp photo** per wall. If it is blurry, take it again. Hold still, tap to focus.
4. About 1 photo in 4 can be a **closer shot** (1 to 2 metres) of one object on the wall, for example just an AC unit or a meter box.
5. Take photos at **different times of day** (morning, noon, evening) and in **sun and shade**. Both help.

## Part 3. What to photograph and how many

The photos must contain real obstacles. One photo can count for several rows.

| What | Aim for at least |
|---|---|
| Walls with windows | 100 |
| Walls with a door or gate | 50 |
| AC units (outdoor unit, split AC, window AC) on a wall | 60 |
| Electricity or water meter boxes on a wall | 40 |
| Pipes running along a wall (drain, rain, gas, water pipes) | 50 |
| Window grills or iron bars | 60 |
| **Clean walls with nothing on them** (important) | 30 |

Mix of wall types: plaster, painted, brick, tile, rough concrete. Mix of places: homes, shops, apartment blocks, college buildings, compound walls.

Also, for **about 20 walls**, do both of these:
- Hold or tape a **measuring tape or a 1 metre ruler** flat on the wall so it shows in the photo.
- Measure the wall's real **width and height** with a tape or laser meter and write them in the notes file (Part 4).

## Part 4. Keep your photos organised

1. Make one folder: `photos_<yourname>` (for example `photos_gopi`). Put every photo in it. Do not rename, crop, edit, filter or compress the photos.
2. In the same folder make a text file `notes.txt`. For each place write one line:
   `place name, kind (home / shop / apartment / college), first and last photo number, measured width x height in metres if you measured`.
   Example: `Anurag canteen, college, IMG_0412 to IMG_0431, 6.2 x 3.1`
3. Delete photos that are blurry, dark, or show the wall at a sharp angle.

## Part 5. Send the photos to Jitesh

1. Zip the folder: right-click > Send to > Compressed folder (Windows), or Share > Compress (phone).
2. Send the zip as a **file / document**, not as normal chat photos. Chat apps shrink photos and the model needs the full size. Best: upload to Google Drive and share the link. WhatsApp: attach as **Document**.
3. In your message, tell Jitesh the total number of photos and how many of each kind from the table.

## Rules about privacy and safety

- Ask permission before photographing a private house or shop. Do not trespass and do not climb anything.
- Avoid people, faces and vehicle number plates. If one is in the photo by accident, retake it.
- Do not post these photos on social media. They are project data.

## Optional A. Check your photos in the project app (needs a laptop)

Only do this if you want to see what the model does with your photos. You need Git and Python tools installed.

1. Install Git and `uv` (https://docs.astral.sh/uv/getting-started/installation/).
2. Get the project: `git clone https://github.com/JITESH-KUMAR05/APR-Sim-mini.git`, then open the folder in a terminal.
3. Set up: `uv sync`
4. Start it: `.venv\Scripts\python.exe app.py`
5. Open http://127.0.0.1:5000 in a browser, click the upload box, choose one of your photos, type the real wall width and height in millimetres, and look at the coloured boxes. Boxes marked "unverified" need your decision. If you change a box, the app keeps your photo and your correction in the `exports/corrections` folder on your laptop. Send that folder to Jitesh too (zip it).

## Optional B. Download the BFA-3D facade dataset (only if you can)

It is a public set of 1,240 facade photos with air conditioner labels, hosted on Baidu Pan, which needs a Baidu account and cannot be downloaded by a script. It may be difficult to register from India; skip this if signup fails.

1. Open https://pan.baidu.com/s/1AiA3nduWC7oJOedhKMxKlQ?pwd=3s2g (the access code `3s2g` is in the link).
2. Sign in or register for a free Baidu account, then save or download the dataset folder. Use the Baidu desktop app if the browser limits the download.
3. Do not unzip it into the project. Zip or share the folder with Jitesh, and write down where you found it. Its licence is not stated: it is for research use only until checked.

## Checklist before you send

- [ ] 200 or more sharp JPG photos, full size, not edited
- [ ] At least 30 clean-wall photos and the numbers in the table covered
- [ ] About 20 walls with a ruler or tape and a measured size in `notes.txt`
- [ ] No faces or number plates
- [ ] Sent as a zip file or document, with the totals in the message
