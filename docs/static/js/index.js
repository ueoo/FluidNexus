window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  // Check for click events on the navbar burger icon

  var options = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 5000,
  };

  // Initialize all div with carousel class
  var carousels = bulmaCarousel.attach(".carousel", options);
  bulmaSlider.attach();
});

// Wait until the DOM is fully loaded (optional but recommended)
document.addEventListener("DOMContentLoaded", () => {
  const toggleResim = document.getElementById("toggle-resim");
  const resimContent = document.getElementById("resim-content");

  // Make sure these elements exist before attaching event listeners
  if (toggleResim && resimContent) {
    toggleResim.addEventListener("click", () => {
      // Toggle the 'display' style
      if (resimContent.style.display === "none" || resimContent.style.display === "") {
        resimContent.style.display = "block";
      } else {
        resimContent.style.display = "none";
      }
    });
  }

  const toggleResimWild = document.getElementById("toggle-resim-wild");
  const resimWildContent = document.getElementById("resim-wild-content");

  // Make sure these elements exist before attaching event listeners
  if (toggleResimWild && resimWildContent) {
    toggleResimWild.addEventListener("click", () => {
      // Toggle the 'display' style
      if (resimWildContent.style.display === "none" || resimWildContent.style.display === "") {
        resimWildContent.style.display = "block";
      } else {
        resimWildContent.style.display = "none";
      }
    });
  }

  // ====== For Ablation Novel View Synthesis ======
  const toggleAblNvs = document.getElementById("toggle-abl-nvs");
  const ablNvsContent = document.getElementById("abl-nvs-content");
  if (toggleAblNvs && ablNvsContent) {
    toggleAblNvs.addEventListener("click", () => {
      // Toggle the display
      if (ablNvsContent.style.display === "none" || ablNvsContent.style.display === "") {
        ablNvsContent.style.display = "block";
      } else {
        ablNvsContent.style.display = "none";
      }
    });
  }

  // ====== For Ablation Re-simulation ======
  const toggleAblResim = document.getElementById("toggle-abl-resim");
  const ablResimContent = document.getElementById("abl-resim-content");
  if (toggleAblResim && ablResimContent) {
    toggleAblResim.addEventListener("click", () => {
      // Toggle the display
      if (ablResimContent.style.display === "none" || ablResimContent.style.display === "") {
        ablResimContent.style.display = "block";
      } else {
        ablResimContent.style.display = "none";
      }
    });
  }
});
