document.addEventListener('DOMContentLoaded', function () {
    // Ensure Bootstrap Tabs Work Correctly (using jQuery for Bootstrap 4.5)
    $('#reportTab a').on('click', function(e) {
      e.preventDefault();
      $(this).tab('show');
    });

    // Form Validation for Complaint Submission
    var form = document.getElementById('description-form');
    var descriptionField = document.getElementById('description');

    if (form && descriptionField) {
      form.addEventListener('submit', function (e) {
        var description = descriptionField.value.trim();
        var wordCount = description.split(/\s+/).filter(function (word) { return word; }).length;
        if (wordCount < 5) {
          e.preventDefault();
          alert('Description must be at least 5 words.\n\nविवरण कम से कम 5 शब्दों का होना चाहिए।');
          return;
        }
        if (!description.includes(' ')) {
          e.preventDefault();
          alert('Please write a proper sentence.\n\nकृपया सही वाक्य लिखें।');
          return;
        }
      });
    }

    // Infinite Scrolling Animation for Instructions
    var scrollContainer = document.querySelector('.scroll-container');
    var scrollContent = document.querySelector('.scroll-content');

    if (scrollContainer && scrollContent) {
      // Clone the scroll content if not already cloned (for seamless looping)
      if (!document.querySelector('.scroll-content.cloned')) {
        var duplicate = scrollContent.cloneNode(true);
        duplicate.classList.add('cloned');
        scrollContainer.appendChild(duplicate);
      }
    }

    // CSV Upload Loading Overlay
    var csvUploadForm = document.getElementById('csv-upload-form');
    if (csvUploadForm) {
      csvUploadForm.addEventListener('submit', function (e) {
        // Show loading overlay
        document.getElementById('loading-overlay').style.display = 'block';
      });
    }

    console.log("JavaScript Loaded Successfully");
  });

  // Hide loading overlay on page load (in case it remains from a previous submission)
  window.addEventListener('load', function () {
    var loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
      loadingOverlay.style.display = 'none';
    }
  });
