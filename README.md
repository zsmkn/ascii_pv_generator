# ascii-pv

Music + lyrics (LRC / SRT / plain text) → ASCII-style PV, played in the terminal or rendered to MP4.

[繁體中文](#繁體中文)

## Setup (Windows 11)

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"          # core
.venv\Scripts\python -m pip install -e ".[audio]"        # M1+: analysis and playback (also needs ffmpeg on PATH)
```

Optional ML layer (vocal separation, lyric auto-sync and plain-text lyric timing, better beats, CLAP style tags). Install the CUDA build of
torch first (RTX 50xx needs CUDA 12.8), then the rest with torch pinned so pip cannot swap it for a CPU build:

```
.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install demucs beat-this transformers --extra-index-url https://download.pytorch.org/whl/cu128
```

`analyze` uses whatever is installed (`--no-ml` to skip it, `--cpu` without a GPU, `--no-sync` to keep the lyric
file's own timing). Models download on first use (~700 MB; plain-text lyrics add ~2.8 GB: the wav2vec2 aligner and
Whisper large-v3-turbo).

## GUI

```
.venv\Scripts\python -m pip install -e ".[gui]"
ascii-pv gui                  # opens http://127.0.0.1:7860 (local only)
```

Four tabs, top to bottom: **Song** (upload music / lyrics / picture, or reopen an analysed song) → **Direct** (seed,
palette, lyric offset; edit shots and lyric cues in tables, lock rows, save) → **Preview** (any frame; play in a
terminal window) → **Render MP4** (size, quality, range, progress bar, result in the page).
Rest the pointer on any button, option or table column title for a short explanation (keyboard focus shows it
too; Esc hides it).
The Song tab's options (ML layer, auto-align, ignore timestamps) apply when a song is analysed. A song reopened
from the list is not analysed again by **Analyse**: to change those options for it, drop its music and lyric files
in again and press **Analyse** (the model results are cached, so this is quick). The lyric offset on the Direct
tab needs no new analysis.

## Usage

```
ascii-pv analyze song.mp3 --lyrics song.lrc          # -> work/<song-id>/analysis.json (cached)
         [--bpm 128] [--lyrics-offset -0.3] [--lyrics-scale 1.0] [--force]
ascii-pv analyze song.mp3 --lyrics song.txt          # plain-text lyrics: timed from the vocals (see Lyrics)
ascii-pv analyze song.mp3 --lyrics bad.lrc --retime  # an LRC whose timing is wrong: keep its text only
ascii-pv info work/<song-id>                         # summary: tempo, style, sections, lyrics, ML, sync
ascii-pv plan work/<song-id> [--seed 3] [--palette synthwave] [--lock] [--image chara.png]
                                                     # direct the PV -> work/<song-id>/storyboard.json
ascii-pv play work/<song-id> [--subtitle]            # play in this terminal (Windows Terminal recommended)
ascii-pv play song.mp3 --lyrics song.lrc             # analyse (cached) and play
         [--start 60] [--fps 30] [--offset 0.2] [--mute] [--stats]
ascii-pv render work/<song-id> [-o out.mp4]          # MP4 (1280x720, 24 fps, song muxed in)
         [--start 60 --end 90] [--crf 23] [--no-post] [--workers 8] [--size 1920x1080 --grid 240x68]
ascii-pv all song.mp3 --lyrics song.lrc [--image chara.png] [-o out.mp4]
                                                     # analyse + plan + play (or render with -o)
ascii-pv preview work/<song-id> --at 64 --png f.png  # one frame as an image
ascii-pv templates                                   # list templates
ascii-pv gui [--port 7860]                           # the web GUI
ascii-pv preview --demo --at 20 --size 120x40        # one frame of a synthetic song
ascii-pv preview work/<song-id> --at 45.2 --plain    # one frame of an analysed song, as plain text
```

## Lyrics

Three kinds of lyric file work:

- **LRC / SRT** (timed). With the ML layer the timing is checked against the separated vocals: a constant offset
  and speed difference (another cut of the song) is found and fixed (`info` shows the `sync` line).
- **Plain text** (`.txt`, or an LRC without timestamps): one lyric line per text line. Blank lines separate
  stanzas; `[Chorus]` / `Verse 2:` headers are skipped; words in parentheses are backing vocals that may or may
  not be sung. The text is timed word by word from the vocals: Whisper says roughly where each word is, a
  wav2vec2 forced aligner says exactly when. Needs the ML layer.
- **`--retime`** (GUI: "ignore the lyric file's timestamps"): treat a timed file as plain text. Use it when the
  LRC's timing drifts (e.g. machine-made LRCs), which a constant offset cannot fix.

Plain-text timing writes `work/<song-id>/lyrics.auto.lrc` (enhanced LRC with word times). Check it, fix a line by
hand if needed, and pass it back with `--lyrics` to use it as is. The text should match what is sung; a few
misspelt words are tolerated, a wrong verse is not. English and Chinese (traditional or simplified, mixed
with English; needs `pypinyin`, part of the `ml` extra); Japanese not yet.

**Chinese / Japanese** lyrics work as timed LRC / SRT (traditional or simplified, mixed with English). Every
character is one sung syllable (and one word for the timing and the lettering); spaces in the line are short
breaths and the preferred places to break a long line. The big letters always use Microsoft JhengHei Bold
(YaHei Bold for characters JhengHei lacks) whatever the cue's `font` says, since thin faces lose their strokes on
the cell grid; textured fills (`binary`, `hex`, `self`) turn solid unless the characters are huge, and the
director never picks `knockout` for them. Plain Chinese text (or `--retime`) is timed from the vocals like English:
both models compare the text as pinyin, so traditional lyrics match what Whisper writes in simplified characters.
Plain Japanese text is not supported yet (`--retime` is ignored for a file with kana).

## Layout

| Path | |
|---|---|
| `ascii_pv/core/` | `Canvas` (cell grid), `Palette`, `Analysis` (analysis.json), `Storyboard` (storyboard.json), `FrameCtx` |
| `ascii_pv/templates/` | background templates; each registers itself with `@register` |
| `ascii_pv/render.py` | per-frame composition shared by every backend |
| `ascii_pv/backends/` | terminal / audio / video output |
| `ascii_pv/analysis/` | ffmpeg decode, tempo/beats/features, lyrics (+ sync, plain-text alignment), sections, style |
| `ascii_pv/typography/` | lyric lettering: bitmap font, ink masks, particle layer |
| `ascii_pv/director/` | the Director: `rules.py` (what to choose), `planner.py` (the planning loop) |
| `ascii_pv/fx/` | post effects (glitch, scan, camera, drop shockwave) and shot transitions |
| `ascii_pv/player.py` | terminal player loop and keys |
| `ascii_pv/imaging.py` | picture -> cells (halfblock / ramp / glyph / edge) and its dance to the music |

Every frame is a pure function of `(t, analysis, storyboard)`; tests check this.

`refs/` holds the two reference projects locally and is not committed.

## Pictures

`--image` takes a character sheet or a cover. A plain background (e.g. white) is cut out automatically, including
holes enclosed by hair or arms; small extras (chibi insets, confetti) are dropped; full / bust / face crops go to
`work/<song-id>/cache/image/`. Without `--image`, cover art embedded in the audio file is used. The Director opens
and closes on the picture, may cut to it mid-song, and puts the character beside the choruses.

## Player keys

SPACE play/pause · ←/→ 5 s · 1–9 sections · `,` `.` previous/next line · `[` `]` lyrics earlier/later ·
`+` `-` volume · R restart · H help · Q quit

## Editing the storyboard

`plan` writes `work/<song-id>/storyboard.json`; `play` and `preview` use it when it exists. Edit it freely:

- `palette`: amber · phosphor · deepsea · synthwave · mono
- `lyric_offset`: seconds for the whole song (+ = lyrics earlier); fixes a late or early LRC once for play and render
- `grid`: character grid of the MP4 (160x45 = 8x16-pixel cells at 1280x720)
- `shots[]`: `start`/`end` (s), `template` (see `ascii-pv templates`), `params`,
  `fx` (0–1 each: `glitch` · `scan` · `zoom` · `shake` · `split` · `strobe` · `noise` ·
  `shockwave` (a drop sends a ring of block glyphs across the screen)),
  `hud` (panels of maths, code and live readouts around the picture, e.g. `["formula:torus", "code:donut",
  "plot:sin", "data"]`: `formula:` pi · trig · euler · torus · fourier · heart · mandel, `plot:` sin · lissajous ·
  signal, `data`, `code:` donut · fft · life · rain · render, `log`, `hex`; `[]` for none).
  `plan` cuts at every drop found by the analysis (`info` lists build / drop / break events); with no lyric on
  screen the drop shot opens on a `shockwave`.
  `spectrum_bars` and `wireframe_mesh` also take `"annotate": false` to drop their analyser markings. 
  `transition_in` (cut · flash · dissolve · crt_on · glitch · wipe · blocks · slide · zoom · reflow)
- `lyrics[]` (one per lyric line):
  - `mode`: takeover (full screen) · line (lower band) · word_flash (one huge word at a time) · stack (words drop in
    as sung) · embed (big letters inside the picture)
  - `font`: bit5x5 · impact · black · serif · script · mono (or a path to any .ttf; Chinese / Japanese lines
    use a CJK face, see Lyrics)
  - `fill`: binary · hex · self (the word's own letters) · ramp · block · outline · solid
  - `enter`: assemble · decode · rain · scan · glitch_in; `exit`: explode · fade · melt · shatter · crt_off
  - `blend`: overlay · mask (the picture shows through the letters) · knockout (letters cut out of the picture) ·
    echo (the previous line lingers) · obstacle (particles flow around the letters)
  - `offset` (s, per line)
- `overlays.sprite`: the character over the choruses — `enabled`, `crop` (full · bust · face), `mode`, `align`
  (left · right · center · alternate), `scale`, `sections`
- `image_ascii` shots: `crop`, `mode` (halfblock · ramp · glyph · edge), `color` (source · palette), `align`,
  `scale`, `reveal` (scan · dissolve · none), `react` (0 = still)
- Set `"locked": true` on any shot or lyric cue you changed, then `ascii-pv plan <work> --lock --seed N` re-rolls
  everything else around it.

---

# 繁體中文

音樂 + 歌詞（LRC / SRT / 純文字）→ ASCII 風格 PV，可在終端機中播放，或輸出成 MP4。

## 安裝（Windows 11）

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"          # 核心
.venv\Scripts\python -m pip install -e ".[audio]"        # M1+：分析與播放（PATH 上也需要 ffmpeg）
```

選用的 ML 層（人聲分離、歌詞自動對時與純文字歌詞計時、更準的節拍、CLAP 風格標籤）。請先安裝 CUDA 版的
torch（RTX 50 系列需要 CUDA 12.8），再在固定 torch 版本的情況下安裝其餘套件，避免 pip 把它換成 CPU 版：

```
.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install demucs beat-this transformers --extra-index-url https://download.pytorch.org/whl/cu128
```

`analyze` 會使用已安裝的任何模型（`--no-ml` 略過 ML 層，沒有 GPU 時用 `--cpu`，`--no-sync` 保留歌詞檔
原本的時間）。模型在第一次使用時下載（約 700 MB；純文字歌詞另需約 2.8 GB：wav2vec2 對齊器與
Whisper large-v3-turbo）。

## GUI

```
.venv\Scripts\python -m pip install -e ".[gui]"
ascii-pv gui                  # 開啟 http://127.0.0.1:7860（僅限本機）
```

由上到下共四個分頁：**Song**（上傳音樂 / 歌詞 / 圖片，或重新開啟已分析的歌曲）→ **Direct**（種子、
配色、歌詞偏移；在表格中編輯鏡頭與歌詞段落、鎖定列、儲存）→ **Preview**（任一畫格；在終端機視窗播放）→
**Render MP4**（尺寸、品質、範圍、進度條，結果直接顯示在頁面上）。
將游標停在任何按鈕、選項或表格欄位標題上，會顯示簡短說明（鍵盤焦點也會顯示；按 Esc 隱藏）。
Song 分頁的選項（ML 層、自動對齊、忽略時間標記）在分析歌曲時生效。從清單重新開啟的歌曲，按 **Analyse**
不會重新分析：若要改變這些選項，請重新拖入它的音樂與歌詞檔再按 **Analyse**（模型結果有快取，所以很快）。
Direct 分頁的歌詞偏移不需要重新分析。

## 使用方式

```
ascii-pv analyze song.mp3 --lyrics song.lrc          # -> work/<song-id>/analysis.json（有快取）
         [--bpm 128] [--lyrics-offset -0.3] [--lyrics-scale 1.0] [--force]
ascii-pv analyze song.mp3 --lyrics song.txt          # 純文字歌詞：依人聲計時（見「歌詞」）
ascii-pv analyze song.mp3 --lyrics bad.lrc --retime  # 時間錯誤的 LRC：只保留文字
ascii-pv info work/<song-id>                         # 摘要：速度、風格、段落、歌詞、ML、對時
ascii-pv plan work/<song-id> [--seed 3] [--palette synthwave] [--lock] [--image chara.png]
                                                     # 編排 PV -> work/<song-id>/storyboard.json
ascii-pv play work/<song-id> [--subtitle]            # 在此終端機播放（建議用 Windows Terminal）
ascii-pv play song.mp3 --lyrics song.lrc             # 分析（有快取）並播放
         [--start 60] [--fps 30] [--offset 0.2] [--mute] [--stats]
ascii-pv render work/<song-id> [-o out.mp4]          # MP4（1280x720、24 fps，混入歌曲音軌）
         [--start 60 --end 90] [--crf 23] [--no-post] [--workers 8] [--size 1920x1080 --grid 240x68]
ascii-pv all song.mp3 --lyrics song.lrc [--image chara.png] [-o out.mp4]
                                                     # 分析 + 編排 + 播放（加 -o 則輸出影片）
ascii-pv preview work/<song-id> --at 64 --png f.png  # 將一個畫格存成圖片
ascii-pv templates                                   # 列出模板
ascii-pv gui [--port 7860]                           # 網頁 GUI
ascii-pv preview --demo --at 20 --size 120x40        # 合成歌曲的一個畫格
ascii-pv preview work/<song-id> --at 45.2 --plain    # 已分析歌曲的一個畫格（純文字）
```

## 歌詞

支援三種歌詞檔：

- **LRC / SRT**（有時間標記）。有 ML 層時，會用分離出的人聲檢查時間：找出並修正固定的偏移與速度差
  （例如歌曲的另一個版本）（`info` 會顯示 `sync` 這一行）。
- **純文字**（`.txt`，或沒有時間標記的 LRC）：每行文字就是一行歌詞。空行分隔段落；`[Chorus]` /
  `Verse 2:` 這類標題會被略過；括號中的字是和聲，可能唱也可能不唱。文字會依人聲逐字計時：Whisper
  估出每個字的大概位置，wav2vec2 強制對齊器再算出精確時間。需要 ML 層。
- **`--retime`**（GUI：「忽略歌詞檔的時間標記」）：把有時間的檔案當成純文字處理。適用於時間會漂移的 LRC
  （例如機器產生的 LRC），這種誤差無法用固定偏移修正。

純文字計時會寫出 `work/<song-id>/lyrics.auto.lrc`（含逐字時間的增強型 LRC）。可以檢查它、必要時手動修正
某一行，再用 `--lyrics` 傳回去原樣使用。文字應與實際演唱內容相符；少數拼錯的字可以容忍，整段歌詞錯了則不行。
支援英文與中文（繁體或簡體，可夾雜英文；需要 `pypinyin`，包含在 `ml` extra 中）；日文尚未支援。

**中文 / 日文**歌詞以有時間的 LRC / SRT 形式皆可使用（繁體或簡體，可夾雜英文）。每個字是一個唱出的音節
（計時與字體排版上也算一個詞）；行中的空格代表短暫換氣，也是長句優先斷行的位置。大字一律使用微軟正黑體
粗體（正黑體缺的字改用微軟雅黑粗體），不論段落的 `font` 設定為何，因為細字體在字元格上會失去筆畫；
紋理填充（`binary`、`hex`、`self`）除非字非常大，否則會變成實心，而且導演不會為它們選擇 `knockout`。
純中文文字（或 `--retime`）會像英文一樣依人聲計時：兩個模型都以拼音比對文字，所以繁體歌詞也能對上
Whisper 以簡體寫出的結果。日文純文字尚未支援（含假名的檔案會忽略 `--retime`）。

## 專案結構

| 路徑 | |
|---|---|
| `ascii_pv/core/` | `Canvas`（字元格）、`Palette`、`Analysis`（analysis.json）、`Storyboard`（storyboard.json）、`FrameCtx` |
| `ascii_pv/templates/` | 背景模板；每個都以 `@register` 註冊自己 |
| `ascii_pv/render.py` | 所有後端共用的逐畫格合成 |
| `ascii_pv/backends/` | 終端機 / 音訊 / 影片輸出 |
| `ascii_pv/analysis/` | ffmpeg 解碼、速度/節拍/特徵、歌詞（+ 對時、純文字對齊）、段落、風格 |
| `ascii_pv/typography/` | 歌詞字體：點陣字型、墨跡遮罩、粒子層 |
| `ascii_pv/director/` | 導演：`rules.py`（選什麼）、`planner.py`（編排迴圈） |
| `ascii_pv/fx/` | 後製特效（glitch、掃描線、鏡頭、drop 衝擊波）與鏡頭轉場 |
| `ascii_pv/player.py` | 終端機播放迴圈與按鍵 |
| `ascii_pv/imaging.py` | 圖片 -> 字元格（halfblock / ramp / glyph / edge）以及隨音樂的律動 |

每個畫格都是 `(t, analysis, storyboard)` 的純函數；測試會檢查這一點。

`refs/` 在本機存放兩個參考專案，不會被 commit。

## 圖片

`--image` 可接受角色設定圖或封面。單色背景（例如白色）會自動去背，包括被頭髮或手臂圍住的空洞；小的
附加物（Q 版小圖、碎紙花）會被移除；全身 / 半身 / 臉部裁切存放在 `work/<song-id>/cache/image/`。
沒有 `--image` 時，會使用音訊檔內嵌的封面。導演會以圖片開場與收尾，可能在歌曲中段切到圖片，並把角色
放在副歌旁邊。

## 播放器按鍵

SPACE 播放/暫停 · ←/→ 5 秒 · 1–9 段落 · `,` `.` 上一行/下一行歌詞 · `[` `]` 歌詞提早/延後 ·
`+` `-` 音量 · R 重新開始 · H 說明 · Q 離開

## 編輯 storyboard

`plan` 會寫出 `work/<song-id>/storyboard.json`；存在時 `play` 與 `preview` 會使用它。可自由編輯：

- `palette`：amber · phosphor · deepsea · synthwave · mono
- `lyric_offset`：整首歌的秒數（+ = 歌詞提早）；一次修正偏晚或偏早的 LRC，播放與輸出都適用
- `grid`：MP4 的字元格（160x45 = 1280x720 下每格 8x16 像素）
- `shots[]`：`start`/`end`（秒）、`template`（見 `ascii-pv templates`）、`params`、
  `fx`（各 0–1：`glitch` · `scan` · `zoom` · `shake` · `split` · `strobe` · `noise` ·
  `shockwave`（drop 時一圈方塊字元掃過畫面）），
  `hud`（圍繞畫面的數學、程式碼與即時數值面板，例如 `["formula:torus", "code:donut",
  "plot:sin", "data"]`：`formula:` pi · trig · euler · torus · fourier · heart · mandel，`plot:` sin · lissajous ·
  signal，`data`，`code:` donut · fft · life · rain · render，`log`，`hex`；`[]` 表示不顯示）。
  `plan` 會在分析找到的每個 drop 切鏡（`info` 會列出 build / drop / break 事件）；畫面上沒有歌詞時，
  drop 鏡頭會以 `shockwave` 開場。
  `spectrum_bars` 與 `wireframe_mesh` 也接受 `"annotate": false`，以去掉分析標記。
  `transition_in`（cut · flash · dissolve · crt_on · glitch · wipe · blocks · slide · zoom · reflow）
- `lyrics[]`（每行歌詞一個）：
  - `mode`：takeover（全螢幕）· line（下方橫帶）· word_flash（一次一個巨大的字）· stack（字隨演唱落下）·
    embed（大字嵌在圖片裡）
  - `font`：bit5x5 · impact · black · serif · script · mono（或任何 .ttf 的路徑；中文 / 日文行使用 CJK
    字型，見「歌詞」）
  - `fill`：binary · hex · self（用該字本身的字母）· ramp · block · outline · solid
  - `enter`：assemble · decode · rain · scan · glitch_in；`exit`：explode · fade · melt · shatter · crt_off
  - `blend`：overlay · mask（圖片從字母中透出）· knockout（字母從圖片中挖空）·
    echo（上一行殘留）· obstacle（粒子繞著字母流動）
  - `offset`（秒，每行各自設定）
- `overlays.sprite`：副歌時疊在畫面上的角色 —— `enabled`、`crop`（full · bust · face）、`mode`、`align`
  （left · right · center · alternate）、`scale`、`sections`
- `image_ascii` 鏡頭：`crop`、`mode`（halfblock · ramp · glyph · edge）、`color`（source · palette）、`align`、
  `scale`、`reveal`（scan · dissolve · none）、`react`（0 = 靜止）
- 在任何你修改過的鏡頭或歌詞段落上設定 `"locked": true`，再執行 `ascii-pv plan <work> --lock --seed N`，
  就會保留它們、重新隨機編排其餘部分。
