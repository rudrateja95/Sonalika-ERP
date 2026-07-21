async function viewOrder(orderNo, btn) {

    const originalHtml = btn.innerHTML;

    btn.disabled = true;
    btn.innerHTML = `
        <span class="spinner-border spinner-border-sm me-1"></span>
        Loading...
    `;

    try {

        const res = await fetch(`/api/order/${orderNo}`);
        const items = await res.json();

        let html = "";

        items.forEach(item => {
            html += `
<tr>

    <td class="text-center align-middle">
        <img src="data:image/jpeg;base64,${item.image}" width="70">
    </td>

    <td class="text-center align-middle">${item.style_no}</td>

    <td class="text-center align-middle">${item.qty}</td>

    <td class="text-center align-middle">${item.gold_color}</td>

    <td class="text-center align-middle">${item.gold_purity}</td>

    <td class="text-center align-middle">${item.diamond_color}</td>

    <td class="text-center align-middle">${item.diamond_clarity}</td>

    <td class="text-center align-middle">${item.remarks}</td>

    <td class="text-center align-middle">${item.net_wt}</td>

    <td class="text-center align-middle">${item.gross_wt}</td>

    <td class="text-center align-middle">
        ${item.gem_chart
                    ? `<img src="data:image/jpeg;base64,${item.gem_chart}" width="70">`
                    : "-"
                }
    </td>

</tr>
`;

        });

        document.getElementById("orderDetails").innerHTML = html;

        new bootstrap.Modal(document.getElementById("orderModal")).show();

    } catch (err) {

        console.error(err);
        alert("Failed to load order.");

    } finally {

        btn.disabled = false;
        btn.innerHTML = originalHtml;

    }
}