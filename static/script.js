const FULL_DASH_ARRAY = 283;
const RESET_DASH_ARRAY = `-57 ${FULL_DASH_ARRAY}`;

//All buttons
let startBtn = document.querySelector(".start");
let stopBtn = document.querySelector(".stop");
let resetBtn = document.querySelector(".reset");

//DOM elements
let timer = document.querySelector("#base-timer-path-remaining");
let timeLabel = document.getElementById("base-timer-label");

//Time related vars
const TIME_LIMIT = 0; //in seconds
let timePassed = 0;
let timeLeft = TIME_LIMIT;
let timerInterval = null;

function reset() {
  clearInterval(timerInterval);
  resetVars();
  startBtn.innerHTML = "Start";
  timer.setAttribute("stroke-dasharray", RESET_DASH_ARRAY);
}

function start(withReset = false) {
  setDisabled(startBtn);
  removeDisabled(stopBtn);
  if (withReset) {
    resetVars();
  }
  startTimer();
}

function stop() {
  setDisabled(stopBtn);
  removeDisabled(startBtn);
  startBtn.innerHTML = "Continue";
  clearInterval(timerInterval);
}

function startTimer() {
  timerInterval = setInterval(() => {
    timePassed = timePassed -= 1;
    timeLeft = TIME_LIMIT - timePassed;
    timeLabel.innerHTML = formatTime(timeLeft);
    setCircleDasharray();

    // if (timeLeft === 0) {
    //   timeIsUp();
    // }
  }, 1000);
}

window.addEventListener("load", () => {
  timeLabel.innerHTML = formatTime(TIME_LIMIT);
  setDisabled(stopBtn);
});

//---------------------------------------------
//HELPER METHODS
//---------------------------------------------
function setDisabled(button) {
  button.setAttribute("disabled", "disabled");
}

function removeDisabled(button) {
  button.removeAttribute("disabled");
}

// function timeIsUp() {
//   setDisabled(startBtn);
//   removeDisabled(stopBtn);
//   clearInterval(timerInterval);
//   let confirmReset = confirm("Time is UP! Wanna restart?");
//   if (confirmReset) {
//     reset();
//     startTimer();
//   } else {
//     reset();
//   }
// }

function resetVars() {
  removeDisabled(startBtn);
  setDisabled(stopBtn);
  timePassed = -1;
  timeLeft = TIME_LIMIT;
  console.log(timePassed, timeLeft);
  timeLabel.innerHTML = formatTime(TIME_LIMIT);
}

function formatTime(time) {
  const minutes = Math.floor(time / 60);
  let seconds = time % 60;

  if (seconds < 10) {
    seconds = `0${seconds}`;
  }

  return `${minutes}:${seconds}`;
}

function calculateTimeFraction() {
  const rawTimeFraction = timeLeft / TIME_LIMIT;
  return rawTimeFraction - (1 / TIME_LIMIT) * (1 - rawTimeFraction);
}

function setCircleDasharray() {
  const circleDasharray = `${(
    calculateTimeFraction() * FULL_DASH_ARRAY
  ).toFixed(0)} 283`;
  console.log("setCircleDashArray: ", circleDasharray);
  timer.setAttribute("stroke-dasharray", circleDasharray);
}



var node = document.getElementById('card');
var btn = document.getElementById('download');
const scale = 750 / node.offsetWidth;

let obj = {
  height: node.offsetHeight * scale,
  width: node.offsetWidth * scale,
  style: {
  transform: "scale(" + scale + ")",
  transformOrigin: "top left",
  width: node.offsetWidth + "px",
  height: node.offsetHeight + "px"
  }
}

btn.onclick = function() {
  domtoimage.toBlob(node, obj)
    .then(function(blob) {
      window.saveAs(blob, 'report.png');
    });
}


function imageloaded() {
  document.getElementById('feed').setAttribute('style', 'display: none');
  document.getElementById('fancy-spinner').setAttribute('style', 'display: none');
  document.getElementById('ring').setAttribute('style', 'display: none');
  document.getElementById('dot').setAttribute('style', 'display: none');
}