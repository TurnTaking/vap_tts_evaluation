const providers = ["Amazon", "Google", "Microsoft", "TCC", "FastPitch"];

async function fetchAudioData() {
  const response = await fetch("audio_data.json");
  const data = await response.json();
  return data;
}

function populateClipTypesSelect(data) {
  const clipTypeSelect = document.getElementById("clip-type");

  for (const type in data) {
    const option = document.createElement("option");
    option.value = type;
    option.textContent = type;
    clipTypeSelect.appendChild(option);
  }
}

function populateFileNamesSelect(data, type) {
  const fileNameSelect = document.getElementById("file-name");
  fileNameSelect.innerHTML = "";

  for (const fileName in data[type]) {
    const option = document.createElement("option");
    option.value = fileName;
    option.textContent = fileName;
    fileNameSelect.appendChild(option);
  }
}

function createAudioPlayer(src, provider) {
  const audioElement = document.createElement("audio");
  audioElement.setAttribute("controls", "");
  audioElement.classList.add("w-full");

  const sourceElement = document.createElement("source");
  sourceElement.setAttribute("src", src);
  sourceElement.setAttribute("type", "audio/wav");

  audioElement.appendChild(sourceElement);

  return audioElement;
}

function displayAudioClips(data, type, fileName) {
  const providers = ["Amazon", "Google", "Microsoft", "TCC", "FastPitch"];

  for (const provider of providers) {
    const column = document.getElementById(provider);
    column.innerHTML = "";

    const subheader = document.createElement("h3");
    subheader.textContent = provider;
    subheader.classList.add("text-lg", "font-semibold", "mb-2");
    column.appendChild(subheader);

    const audioFile = data[type][fileName][provider];
    const audioPlayer = createAudioPlayer(
      `data/audio/${type}/${audioFile}`,
      provider
    );
    column.appendChild(audioPlayer);
  }
}

async function main() {
  const data = await fetchAudioData();

  populateClipTypesSelect(data);
  populateFileNamesSelect(data, Object.keys(data)[0]);

  const clipTypeSelect = document.getElementById("clip-type");
  const fileNameSelect = document.getElementById("file-name");

  clipTypeSelect.addEventListener("change", (event) => {
    const type = event.target.value;
    populateFileNamesSelect(data, type);
    displayAudioClips(data, type, fileNameSelect.value);
  });

  fileNameSelect.addEventListener("change", (event) => {
    const fileName = event.target.value;
    displayAudioClips(data, clipTypeSelect.value, fileName);
  });

  displayAudioClips(data, clipTypeSelect.value, fileNameSelect.value);
}

main();
