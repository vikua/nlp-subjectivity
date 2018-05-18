$(function() {
    $('#classify-button').click(function () {
        $.ajax({
            url: "/classify",
            type: "POST",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({"text": $('#article').val()}),
            processData: false,
            success: function (response) {
                var text = "";
                for (i = 0; i < response.length; i++) {

                    let score = response[i]["score"];
                    let tag = "<mark data-toggle='tooltip' data-placement='top'";
                    if (score < 0) {
                        score = -score;
                        tag += "class='bg-danger' style='opacity: " + score + "'";
                    } else {
                        tag += "class='bg-success' style='opacity: " + score + "'"
                    }
                    tag += " title='" + score + "'>";

                    text += tag + response[i]["sentence"] + "</mark>";
                    if (i < response.length - 1) {
                        text += " "
                    }
                }
                $("div#result").show();
                $("div#result div.form-control").html(text);

                $('[data-toggle="tooltip"]').tooltip();
            },
            error: function (response) {
                console.log(response);
            }
        });
    });
});


function applyHighlights(text) {
  text = text
    .replace(/\n$/g, '\n\n')
    .replace(/[A-Z].*?\b/g, '<mark>$&</mark>');
  return text;
}

