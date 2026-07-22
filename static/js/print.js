document.addEventListener("DOMContentLoaded", function () {

    window.printOrder = function () {

        const printArea = document.getElementById("printArea");

        if (!printArea) {
            alert("Print area not found.");
            return;
        }

        const printContents = printArea.innerHTML;

        const printWindow = window.open("", "", "width=1200,height=800");

        printWindow.document.write(`
            <html>
            <head>
                <title>Order details</title>
            </head>
            <body>
                ${printContents}
            </body>
            </html>
        `);

        printWindow.document.close();
        printWindow.print();

    };

});